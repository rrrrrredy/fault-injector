"""CPU-only reward and post-verification export for the four frozen functional test slices."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .sandbox import Pool, commit, execute, git, guard_mounts, write_files, TRAIN_REVISION

ROOT = Path(__file__).resolve().parents[1]
HASH_SEEDS = ("1729", "2027", "4093")


def extract_diff(completion):
    if not isinstance(completion, str) or len(completion) > 128 * 1024:
        return ""
    # A unified diff can end in a single-space context line. Stripping it corrupts the hunk.
    text = completion.replace("\r\n", "\n").lstrip()
    if text.startswith("<think>"):
        _, end, text = text.partition("</think>")
        if not end:
            return ""
        text = text.lstrip()
    if text.startswith("\x60\x60\x60"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or lines[0] not in ("\x60\x60\x60diff", "\x60\x60\x60patch", "\x60\x60\x60")
            or lines[-1] != "\x60\x60\x60"
        ):
            return ""
        text = "\n".join(lines[1:-1])
    if not text.startswith(("diff --git ", "--- ")):
        return ""
    return text.rstrip("\r\n") + "\n"


def reward(observation):
    """Deterministic, side-effect-free score. Infrastructure errors cannot earn positive credit."""
    if (
        observation["reason"] == "verified"
        and observation["failed_nodes"]
        and observation["reverse_passed"]
        and observation["reset_matches"]
        and observation["faulty_exit_code"] == 1
        and observation["fixed_exit_code"] == 0
    ):
        return 1.0
    if observation["reason"] == "pass":
        return 0.0
    return -1.0


def evaluate(pool, seed_id, completion, *, timeout=None):
    result = pool.verify(seed_id, extract_diff(completion), timeout=timeout)
    result["reward"] = reward(result)
    return result


def prepare_export(tools):
    from replay import prepare_tools

    tools = Path(tools).resolve()
    tools.mkdir(parents=True, exist_ok=True)
    pin = (Path(__file__).with_name("requirements-cpu.txt")).read_text(encoding="utf-8").splitlines()
    prepare_tools(tools, (next(line for line in pin if line.startswith("wasmtime==")),))


def export_case(train, seed_id, completion, *, runtime, tools, out):
    """Repeat verification, materialize base/two variants with Factory, validate and return the receipt."""
    out, tools = Path(out).resolve(), Path(tools).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Export needs an empty destination")
    lock = json.loads((ROOT / "stage0.lock.json").read_bytes())
    binary = tools / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    cli = tools / "cli" / ("rci.cmd" if os.name == "nt" else "bin/rci")
    factory = (
        tools
        / ("workflow-environment-factory-" + lock["upstreams"]["workflow-environment-factory"])
        / "backend"
    )
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(factory), str(ROOT)]),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
    )
    argv = [
        str(binary),
        "-m",
        "stage2.rewards",
        "_export",
        str(Path(train).resolve()),
        seed_id,
        str(Path(runtime).resolve()),
        str(cli),
        str(out),
    ]
    subprocess.run(argv, input=completion.encode("utf-8"), cwd=ROOT, env=env, check=True)
    return json.loads((out / "receipt.json").read_bytes())


def _export(train, seed_id, runtime, cli, out, completion):
    from materialize import EvidenceFactory, write_json
    from workflow_environment_factory.content_store import ContentStore
    from workflow_environment_factory.engine import ProcessResult
    from workflow_environment_factory.gitops import GitWorkspaceManager
    from workflow_environment_factory.models import BlueprintCreate
    from workflow_environment_factory.protocol import ProtocolValidator
    from workflow_environment_factory.store import FactoryStore

    with Pool(train, runtime=runtime, workers=1) as pool:
        observation = evaluate(pool, seed_id, completion)
        seed = pool.seeds[seed_id]
    if observation["reward"] <= 0:
        raise ValueError(f"Cannot export unverified mutation: {observation['reason']}")
    delta = extract_diff(completion).encode()
    nodes = observation["failed_nodes"]
    argv = ["python", "-B", "-m", "pytest", "-q", "-o", "pythonpath=" + seed.import_root, *nodes]
    source = hashlib.sha256(seed_id.encode() + b"\x00" + delta).hexdigest()
    lock = json.loads((ROOT / "stage0.lock.json").read_bytes())
    observations = {}

    class GuestEngine:
        name = "wasi-cpython-pytest"

        def run(self, workspace, image, command, timeout_ms):
            guard_mounts()
            if list(command) != argv:
                raise ValueError("Unexpected verifier command")
            value = (workspace / ".fi-hash-seed").read_text().strip()
            if value not in HASH_SEEDS:
                raise ValueError("Unexpected verifier hash seed")
            start = time.monotonic()
            result = execute(
                runtime, workspace, checks, seed, timeout=timeout_ms / 1000, hash_seed=value, nodes=nodes
            )
            observations[workspace.name] = result
            status = {"pass": "pass", "assertion": "fail"}.get(result["status"], "error")
            return ProcessResult(
                status,
                result.get("pytest_exit"),
                json.dumps(result),
                "",
                round((time.monotonic() - start) * 1000),
            )

    with tempfile.TemporaryDirectory(prefix="fi-case-") as temp:
        work = Path(temp).resolve()
        repo, checks = work / "repo", work / "checks"
        write_files(repo, dict(seed.files, **seed.checks))
        write_files(checks, seed.checks)
        (repo / ".fi-hash-seed").write_text(HASH_SEEDS[0] + "\n", encoding="ascii", newline="\n")
        git(repo, "init", "--quiet")
        commit(repo, "Frozen healthy slice")
        git(repo, "apply", "--index", "-", data=delta)
        fault = commit(repo, "Candidate functional regression")
        git(repo, "apply", "--index", "--reverse", "-", data=delta)
        fixed = commit(repo, "Restore frozen behavior")
        payload = BlueprintCreate(
            name=f"stage2-{seed_id}-{source[:12]}",
            kind="code",
            repository_path=str(repo),
            base_revision=fault,
            solution_revision=fixed,
            external_ref="https://huggingface.co/datasets/RedinGhost/fi-python-seeds-v1/tree/"
            + TRAIN_REVISION
            + "/train",
            title_template="Reproduce the " + seed_id + " functional regression with hash seed {value}",
            goal_template="Restore the frozen assertions with PYTHONHASHSEED={value}.",
            completion_summary="The selected original test nodes pass.",
            variable={
                "name": "python_hash_seed",
                "original": HASH_SEEDS[0],
                "variants": list(HASH_SEEDS[1:]),
                "paths": [".fi-hash-seed"],
                "confirmed_by_user": True,
                "description": "Set the interpreter hash seed before the same target assertions.",
            },
            container_image=lock["blueprint_image"],
            verifier={"argv": argv, "timeout_ms": 15000},
            allowed_paths=observation["paths"],
        )
        protocol = ProtocolValidator(ROOT / "schemas", semantic=True)
        with closing(FactoryStore(work / "factory.sqlite")) as store:
            factory = EvidenceFactory(
                store=store,
                content_store=ContentStore(work / "objects"),
                protocol=protocol,
                git=GitWorkspaceManager(work / "worktrees"),
                engine=GuestEngine(),
            )
            blueprint = factory.create_blueprint(payload)
            cases = factory.generate_cases(blueprint.blueprint_id)
        if [case.variant_index for case in cases] != [0, 1, 2]:
            raise RuntimeError("Factory case lineage incomplete")
        documents = out / "cases"
        documents.mkdir()
        rows = []
        for case in cases:
            index, val = case.variant_index, case.validation
            prefix = f"case-{index}"
            if not val.objective_gate_passed:
                raise RuntimeError(f"{prefix} Factory gate failed: {val} {observations}")
            first, second = [
                factory.fingerprints[prefix + "-baseline-" + suffix] for suffix in ("a", "b")
            ]
            before, after = observations[prefix + "-baseline-b"], observations[prefix + "-solution"]
            if not (
                first == second == val.baseline_manifest_ref
                and val.reset_verified
                and before["failed_nodes"] == nodes
                and after["status"] == "pass"
                and before["hash_seed"] == after["hash_seed"] == case.variable_value
                and before["hash_probe"] == after["hash_probe"]
            ):
                raise RuntimeError(f"{prefix} inconsistent reset, failure or interpreter control")
            doc = case.protocol_case
            if doc["provenance"].get("parent_case_id") != (str(cases[0].case_id) if index else None):
                raise RuntimeError("Factory parent does not match the base case")
            doc["extensions"]["fault_injector"] = {
                "source_id": source,
                "seed_id": seed_id,
                "hf_revision": TRAIN_REVISION,
                "hash_seed": case.variable_value,
                "execution_engine": GuestEngine.name,
            }
            doc["environment"]["summary"] = (
                "Frozen functional fixture evaluated with capability-isolated WASI CPython 3.13.7."
            )
            protocol.validate(doc)
            write_json(documents / (prefix + ".json"), doc)
            rows.append(
                dict(
                    variant_index=index,
                    case_id=str(case.case_id),
                    seed=case.variable_value,
                    objective_gate_passed=True,
                    reset_verified=val.reset_verified,
                    reset_fingerprint_1=first,
                    reset_fingerprint_2=second,
                    baseline_fingerprint=val.baseline_manifest_ref,
                    faulty=before,
                    fixed=after,
                )
            )
        if len({r["faulty"]["hash_probe"] for r in rows}) != 3:
            raise RuntimeError("Hash-seed controls did not affect the guest interpreter")
        # Preserve the exact Git objects referenced by the exported documents.
        git(repo, "bundle", "create", str(out / "fixture.bundle"), "HEAD")
        git(repo, "bundle", "verify", str(out / "fixture.bundle"))
    checked = subprocess.run(
        [str(cli), "validate", "--semantic", "cases"],
        cwd=out,
        text=True,
        encoding="utf-8",
        capture_output=True,
    )
    if checked.returncode:
        raise RuntimeError(checked.stdout + "\n" + checked.stderr)
    (out / "candidate.diff").write_bytes(delta)
    write_json(
        out / "receipt.json",
        dict(
            seed_id=seed_id,
            hf_revision=TRAIN_REVISION,
            source_id=source,
            source_count=1,
            faulty_commit=fault,
            fixed_commit=fixed,
            target_nodes=nodes,
            cases=rows,
            rci_exit=checked.returncode,
            rci_stdout=checked.stdout,
            rci_stderr=checked.stderr,
        ),
    )
    print(checked.stdout, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_export"]:
        _export(
            Path(sys.argv[2]),
            sys.argv[3],
            Path(sys.argv[4]),
            Path(sys.argv[5]),
            Path(sys.argv[6]),
            sys.stdin.buffer.read().decode("utf-8"),
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("command", choices=["prepare-export"])
        parser.add_argument("--tools", type=Path, required=True)
        prepare_export(parser.parse_args().tools)
