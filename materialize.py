"""Run the trusted fixture through the pinned Factory and retain its actual observations."""

from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import sys

from workflow_environment_factory.content_store import ContentStore
from workflow_environment_factory.engine import LocalTestEngine
from workflow_environment_factory.factory import CaseFactory
from workflow_environment_factory.gitops import GitWorkspaceManager
from workflow_environment_factory.models import BlueprintCreate
from workflow_environment_factory.protocol import ProtocolValidator
from workflow_environment_factory.store import FactoryStore

ROOT = Path(__file__).resolve().parent


class EvidenceEngine(LocalTestEngine):
    def __init__(self):
        self.invocations = {}

    def run(self, workspace, image, argv, timeout_ms):
        # Windows can find the launcher's Python ahead of PATH. Keep the exported command portable.
        if argv[0] != "python":
            raise ValueError("The stage 0 fixture requires the replay's Python interpreter")
        result = super().run(workspace, image, [sys.executable, *argv[1:]], timeout_ms)
        self.invocations[workspace.name] = asdict(result)
        return result


class EvidenceFactory(CaseFactory):
    def __init__(self, **dependencies):
        super().__init__(**dependencies)
        self.fingerprints = {}

    def _workspace_state_digest(self, workspace):
        reference = super()._workspace_state_digest(workspace)
        self.fingerprints[workspace.name] = reference
        return reference


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def materialize(work, evidence, cli, source_file=None):
    lock = json.loads((ROOT / "stage0.lock.json").read_text(encoding="utf-8"))
    source = json.loads(source_file.read_text(encoding="utf-8")) if source_file else lock["source"]
    repository = work / Path(source["repository"]).stem
    spec = source.get("replay", {
        "title": "Fractional bytes retain digits in human-readable file sizes",
        "goal": "Restore the integer byte count for naturalsize(1.123456789) using default precision {value}.",
        "completion": "The reported input renders as 1 Bytes.",
        "failure": "assert '1.123456789 Bytes' == '1 Bytes'",
    })
    fixture, test_name = source["nodeid"].split("::")
    target = f"{Path(fixture).stem}::{test_name}"
    verifier = ["python", "-B", "-m", "pytest", "-q", "-s", "--tb=short", "--color=no",
                "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", "-o", "filterwarnings=",
                "--pyargs", target]
    payload = BlueprintCreate(
        name=source["id"], kind="code", repository_path=str(repository),
        base_revision=source["fault"], solution_revision=source["fix"], external_ref=source["issue"],
        title_template=spec["title"], goal_template=spec["goal"], completion_summary=spec["completion"],
        variable=source["variable"], container_image=lock["blueprint_image"],
        verifier={"argv": verifier, "timeout_ms": 30000}, allowed_paths=source["variable"]["paths"],
    )
    protocol = ProtocolValidator(ROOT / "schemas", semantic=True)
    engine = EvidenceEngine()
    with closing(FactoryStore(work / "factory.sqlite")) as store:
        factory = EvidenceFactory(store=store, content_store=ContentStore(work / "content"),
                                  protocol=protocol, git=GitWorkspaceManager(work / "worktrees"), engine=engine)
        blueprint = factory.create_blueprint(payload)
        cases = factory.generate_cases(blueprint.blueprint_id)

    case_dir = evidence / "cases"
    case_dir.mkdir()
    if [case.variant_index for case in cases] != [0, 1, 2]:
        raise ValueError("Factory did not return the expected base and two derived cases")
    base_id = str(cases[0].case_id)
    rows = []
    for case in cases:
        index = case.variant_index
        prefix = f"case-{index}"
        validation = case.validation
        if not validation.objective_gate_passed:
            failure = evidence / "failed-gate.json"
            write_json(failure, {"case": prefix, "validation": validation.model_dump(mode="json"),
                                 "invocations": {k: v for k, v in engine.invocations.items() if k.startswith(prefix)}})
            raise RuntimeError(f"{prefix} did not pass the Factory gate: {validation.details} See {failure}")
        first = factory.fingerprints[f"{prefix}-baseline-a"]
        second = factory.fingerprints[f"{prefix}-baseline-b"]
        if not validation.reset_verified or not (first == second == validation.baseline_manifest_ref):
            raise RuntimeError(f"{prefix} reset fingerprints disagree with the Factory baseline")
        before = engine.invocations[f"{prefix}-baseline-b"]
        after = engine.invocations[f"{prefix}-solution"]
        if (before["exit_code"] != 1 or spec["failure"] not in before["stdout"]
                or after["exit_code"] != 0 or "1 passed" not in after["stdout"]):
            raise RuntimeError(f"{prefix} did not show the specific failing/passing assertion: {before} {after}")
        if "control" in spec:
            control = spec["control"].format(value=case.variable_value)
        else:
            control = f"precision control (3000 bytes): {case.variable_value % 3} kB"
        if any(control not in result["stdout"] for result in (before, after)):
            raise RuntimeError(f"{prefix} did not exhibit the declared control: {control}")
        doc = case.protocol_case
        parent = doc["provenance"].get("parent_case_id")
        if parent != (base_id if index else None):
            raise ValueError(f"{prefix} has an incorrect parent case")
        doc["extensions"]["fault_injector"] = {"regression_source_id": source["id"]}
        doc["environment"]["summary"] = f"Trusted {repository.name} source fixture replayed with Factory LocalTestEngine on the host."
        protocol.validate(doc)
        write_json(case_dir / f"{prefix}.json", doc)
        rows.append({
            "case_id": str(case.case_id), "variant_index": index, "value": case.variable_value,
            "regression_source_id": doc["extensions"]["fault_injector"]["regression_source_id"],
            "parent_case_id": parent, "validation": validation.model_dump(mode="json"),
            "reset_fingerprint_1": first, "reset_fingerprint_2": second,
            "faulty": before, "fixed": after,
        })
        print(f"{prefix} {case.variable_value} | {control} | objective_gate_passed=true")
    count = len({row["regression_source_id"] for row in rows})
    if count != 1:
        raise ValueError(f"Expected one regression source, got {count}")
    checked = subprocess.run([str(cli), "validate", "--semantic", "cases"], cwd=evidence,
                             capture_output=True, text=True, encoding="utf-8")
    print(checked.stdout, end="")
    if checked.stderr:
        print(checked.stderr, file=sys.stderr)
    checked.check_returncode()
    base_tree = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", source["fault"] + "^{tree}"], text=True,
    ).strip()
    receipt = {
        "project_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source": source, "upstreams": lock["upstreams"], "host": platform.system(),
        "python": platform.python_version(), "execution_engine": engine.name,
        "declared_blueprint_image": lock["blueprint_image"], "baseline_git_tree": base_tree,
        "source_count": count, "cases": rows,
        "rci": {"command": "rci validate --semantic cases", "exit_code": checked.returncode,
                "stdout": checked.stdout, "stderr": checked.stderr},
    }
    write_json(evidence / "receipt.json", receipt)
    print(rows[0]["faulty"]["stdout"], rows[0]["fixed"]["stdout"], sep="\n")
    print(f"source_count={count} | baseline_git_tree={base_tree}")
    print(f"reset_fingerprint_1={rows[0]['reset_fingerprint_1']}")
    print(f"reset_fingerprint_2={rows[0]['reset_fingerprint_2']}")
    print(f"baseline_manifest_ref={cases[0].validation.baseline_manifest_ref} | reset_verified=true")
    print(f"rci exit={checked.returncode} | receipt={evidence / 'receipt.json'}")


if __name__ == "__main__":
    materialize(*(Path(argument) for argument in sys.argv[1:]))
