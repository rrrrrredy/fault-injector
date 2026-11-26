"""Freeze and independently verify small Python corpora. Run with CPython 3.12 and Node/npm."""
import argparse
import ast
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from corpus import blob, check_ticket, get_repo, git, group_families, inspect_source, json_bytes as encode, sha, slug

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from replay import checkout_regression, prepare_tools


def files_digest(files):
    return sha(encode({name: sha(data) for name, data in sorted(files.items())}))


def write_files(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def tree_files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and ".git" not in p.relative_to(root).parts}


def zip_files(path, files):
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for name, data in sorted(files.items()):
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def finish_package(path, card):
    (path / "README.md").write_bytes((HERE / card).read_bytes())
    for name in ("LICENSE", "NOTICE"):
        (path / name).write_bytes((ROOT / name).read_bytes())
    files = tree_files(path)
    checksum = "".join(f"{sha(data)}  {name}\n" for name, data in sorted(files.items()))
    (path / "SHA256SUMS").write_bytes(checksum.encode())
    transport = path.with_suffix(".zip")
    zip_files(transport, tree_files(path))
    return sha(transport.read_bytes())


def load_tokenizer(cache, lock):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    for name, digest in lock["files"].items():
        path = cache / name
        if not path.exists():
            url = f"https://huggingface.co/{lock['repository']}/resolve/{lock['commit_sha']}/{name}"
            with urlopen(url, timeout=90) as response:
                data = response.read()
            if sha(data) != digest:
                raise ValueError(f"Unexpected tokenizer download: {name}")
            path.write_bytes(data)
        if sha(path.read_bytes()) != digest:
            raise ValueError(f"Tokenizer checksum mismatch: {name}")
    config = json.loads((cache / "tokenizer_config.json").read_bytes())
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(config["chat_template"])
    return Tokenizer.from_file(str(cache / "tokenizer.json")), template


def selected_test(row, visibility):
    node = row[visibility + "_node"]
    path, name = node.split("::")
    source = (HERE / (visibility + "_tests.py")).read_text(encoding="utf-8")
    function, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    return path, (ast.get_source_segment(source, function) + "\n").encode()


def healthy_files(repo, row, tokenizer, template):
    files = {p: blob(repo, row["commit_sha"], p) for p in row["files"]}
    public, public_data = selected_test(row, "public")
    private, private_data = selected_test(row, "oracle")
    if public == private or public_data == private_data or row["public_node"].split("::")[1] == row["oracle_node"].split("::")[1]:
        raise ValueError("Public and oracle tests overlap")
    files[public] = public_data
    task = {"task": row["task"]}
    content = row["task"] + "\n" + "".join(f"\n--- {name} ---\n{data.decode('utf-8')}"
                                            for name, data in sorted(files.items()))
    # This exact public input is shipped; the published token count includes the generation prefix.
    rendered = template.render(messages=[{"role": "user", "content": content}], add_generation_prompt=True)
    count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
    if count > 9500:
        raise ValueError(f"{row['slice_id']} exceeds the 9,500-token limit: {count}")
    files["task.json"] = encode(task)
    files["prompt.txt"] = rendered.encode()
    files["licenses/" + row["notice"]] = blob(repo, row["commit_sha"], row["notice"])
    oracle = {private: private_data}
    metrics = dict(tokens=count, prompt_sha256=sha(files["prompt.txt"]), inputs_sha256=files_digest(files),
                   public_node=row["public_node"], oracle_node=row["oracle_node"], oracle_sha256=sha(private_data),
                   input_files={name: sha(data) for name, data in sorted(files.items())})
    return files, oracle, metrics


def suite(worktree, row, node):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    args = [sys.executable, "-B", str(Path(__file__)), "_suite", row["import_root"], row["module"], node]
    start = time.perf_counter()
    result = subprocess.run(args, cwd=worktree, env=env, text=True, encoding="utf-8",
                            capture_output=True, timeout=20)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{node}\n{result.stdout}\n{result.stderr}")
    data = json.loads(result.stdout.splitlines()[-1])
    if data["passed"] != [node] or not data["business_functions"]:
        raise ValueError(f"{node} did not run against the frozen business module: {data}")
    return dict(data, seconds=round(elapsed, 4))


def run_suite(import_root, module, node):
    import pytest

    base = Path.cwd()
    sys.path.insert(0, str((base / import_root).resolve()))
    target = str((base / module).resolve())
    funcs = set()
    passed = []

    class Observations:
        def pytest_runtest_logreport(self, report):
            if report.when == "call" and report.passed:
                passed.append(report.nodeid)

    def enter(code, offset):
        if code.co_filename == target and not code.co_name.startswith("<"):
            funcs.add(code.co_name)

    monitor = sys.monitoring
    tool_id = monitor.PROFILER_ID
    monitor.use_tool_id(tool_id, "slice business calls")
    monitor.register_callback(tool_id, monitor.events.PY_START, enter)
    monitor.set_events(tool_id, monitor.events.PY_START)
    try:
        result = pytest.main(["-q", "--color=no", "--noconftest", "-p", "no:cacheprovider",
                              "--rootdir=.", node], plugins=[Observations()])
    finally:
        monitor.set_events(tool_id, 0)
        monitor.free_tool_id(tool_id)
    print(json.dumps(dict(passed=passed, business_functions=sorted(funcs))))
    return result


def verify_seed(work, row, files, oracle, metrics):
    original = work / "original"
    write_files(original, files)
    git(original, "init", "-q")
    git(original, "config", "core.autocrlf", "false")
    git(original, "add", "--force", "--", ".")
    git(original, "-c", "user.name=Corpus", "-c", "user.email=corpus@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "Frozen healthy slice")
    observations = {}
    fingerprints = []
    for side in ("public", "oracle"):
        checkout = work / side
        git(original, "worktree", "add", "--quiet", "--detach", str(checkout), "HEAD")
        # Exercise tracked restoration and untracked cleanup without editing any business code.
        (checkout / "task.json").unlink()
        (checkout / "disposable.tmp").write_bytes(b"reset probe")
        git(checkout, "reset", "--hard", "--quiet", "HEAD")
        git(checkout, "clean", "-ffdqx")
        fingerprint = files_digest(tree_files(checkout))
        if fingerprint != metrics["inputs_sha256"]:
            raise ValueError(f"{row['slice_id']} reset {side} changed frozen bytes")
        fingerprints.append(fingerprint)
        if side == "oracle":
            (checkout / row["public_node"].split("::")[0]).unlink()
            (checkout / "prompt.txt").unlink()
            write_files(checkout, oracle)
        observations[side] = suite(checkout, row, row[side + "_node"])
    common = sorted(set(observations["public"]["business_functions"]) & set(observations["oracle"]["business_functions"]))
    if not common:
        raise ValueError(f"Suites did not exercise a common business function: {row['module']}")
    seconds = round(sum(x["seconds"] for x in observations.values()), 4)
    if seconds >= 10:
        raise ValueError(f"{row['slice_id']} exceeds the seconds-level verification budget ({seconds}s)")
    overlap = sorted(set(observations["public"]["passed"]) & set(observations["oracle"]["passed"]))
    if overlap:
        raise ValueError(f"Shared test node: {overlap}")
    return dict(observations, common_business_functions=common, module=row["module"], seconds=seconds,
                reset_fingerprint_1=fingerprints[0], reset_fingerprint_2=fingerprints[1],
                baseline_fingerprint=metrics["inputs_sha256"], overlap_nodes=overlap, tokens=metrics["tokens"])


def reference_material(repo, row):
    reference = row["reference"]
    source = (json.loads((ROOT / "stage0.lock.json").read_bytes())["source"]
              if reference.get("source_from_stage0") else reference["source"])
    if source["repository"] != row["repository"] or source["fix"] != row["commit_sha"]:
        raise ValueError(f"Inconsistent reference provenance: {slug(row)}")
    if git(repo, "rev-parse", source["fix"] + "^").decode().strip() != source["fault"]:
        raise ValueError(f"The frozen faulty state is not the parent of the fix: {source['id']}")
    git(repo, "merge-base", "--is-ancestor", source["fix"], reference["merged_sha"])
    ticket = reference["ticket"]
    check_ticket(ticket)
    actor = {"ticket.json": encode(ticket), "licenses/" + row["notice"]: blob(repo, source["fault"], row["notice"])}
    grader = {"provenance.json": encode(dict(source, merged_sha=reference["merged_sha"])),
              "licenses/" + row["notice"]: blob(repo, source["fix"], row["notice"])}
    for path in source["variable"]["paths"]:
        actor["workspace/" + path] = blob(repo, source["fault"], path)
        grader["fixed/" + path] = blob(repo, source["fix"], path)
    fixture = source["nodeid"].split("::")[0]
    grader[fixture] = (ROOT / fixture).read_bytes()
    return source, actor, grader


def replay_reference(work, evidence, cli, repo, source):
    evidence.mkdir(parents=True)
    work.mkdir()
    source_file = work / "source.json"
    source_file.write_bytes(encode(source))
    local = work / Path(source["repository"]).stem
    checkout_regression(local, repo.as_uri(), source["fix"])
    subprocess.run([sys.executable, str(ROOT / "materialize.py"), str(work), str(evidence), str(cli),
                    str(source_file)], check=True)
    receipt = json.loads((evidence / "receipt.json").read_bytes())
    base = receipt["cases"][0]
    return dict(source=source["id"], fault=source["fault"], fix=source["fix"], node=source["nodeid"],
                source_count=receipt["source_count"], variants=[item["variant_index"] for item in receipt["cases"]],
                reset_fingerprint_1=base["reset_fingerprint_1"], reset_fingerprint_2=base["reset_fingerprint_2"],
                baseline_fingerprint=base["validation"]["baseline_manifest_ref"],
                reset_verified=base["validation"]["reset_verified"], gate=base["validation"]["objective_gate_passed"],
                faulty_exit=base["faulty"]["exit_code"], fixed_exit=base["fixed"]["exit_code"],
                rci=receipt["rci"]["exit_code"], rci_stdout=receipt["rci"]["stdout"])


def add_members(bundle, key, files):
    for name, data in files.items():
        destination = key + "/" + name
        if destination in bundle:
            raise ValueError(f"Duplicate release member: {destination}")
        bundle[destination] = data


def build(work, cache, out, cli, freeze):
    manifest = json.loads((HERE / "manifest.json").read_bytes())
    rows = manifest["repositories"]
    tokenizer, template = load_tokenizer(cache, manifest["tokenizer"])
    repos, facts, prints = {}, {}, {}
    for row in rows:
        name = slug(row)
        repos[name] = get_repo(cache, row)
        facts[name], prints[name] = inspect_source(repos[name], row)
    families, comparisons, overlaps = group_families(rows, facts, prints)
    matched = [pair for pair in comparisons if pair["reasons"]]
    fork = {"rspeer/ordered-set", "idanmiara/ordered-set"}
    if not any({p["left"], p["right"]} == fork and "shared_git_root" in p["reasons"] for p in matched):
        raise ValueError("The real renamed cross-owner fork was not detected")
    if next(r["split"] for r in rows if slug(r) == "python-humanize/humanize") != "human_regressions":
        raise ValueError("The stage 0 family must stay on the reference side")

    seed_dir = out / "release" / "fi-python-seeds-v1"
    reference_dir = out / "release" / "human-regressions-v1"
    groups = {s: {"actor": {}, "grader": {}} for s in ("train", "eval_unseen", "human_regressions")}
    seeds, references, healthy_results, reference_results = {}, {}, {}, {}
    for row in rows:
        name, side = slug(row), row["split"]
        if side == "fork_control":
            continue
        repo = repos[name]
        grader_info = {k: row[k] for k in ("repository", "commit_sha", "spdx", "notice")}
        grader_info.update(facts[name], repo_family_id=families[name])
        if side == "human_regressions":
            key = row["reference"]["blind_id"]
            source, actor, grader = reference_material(repo, row)
            references[key] = dict(source=source["id"], family=families[name], actor_sha256=files_digest(actor),
                                   grader_sha256=files_digest(grader), ticket_sha256=sha(actor["ticket.json"]))
            if not freeze:
                reference_results[key] = replay_reference(work / key, out / "references" / key, cli, repo, source)
                reference_results[key].update(ticket_scan_hits=0, actor_paths=sorted(actor))
                print(f"{key} | source={source['id']} | counted once | ticket scan passed", flush=True)
        else:
            key = row["slice_id"]
            actor, grader, metrics = healthy_files(repo, row, tokenizer, template)
            seeds[key] = dict(metrics, repo_family_id=families[name], split=side)
            grader_info.update(metrics, module=row["module"], import_root=row["import_root"])
            if not freeze:
                healthy_results[key] = verify_seed(work / key, row, actor, grader, metrics)
                measured = healthy_results[key]
                print(f"{side}/{key} | 1 public + 1 oracle passed | {metrics['tokens']} tokens | "
                      f"{measured['seconds']:.3f}s | resets equal", flush=True)
        grader["source.json"] = encode(grader_info)
        add_members(groups[side]["actor"], key, actor)
        add_members(groups[side]["grader"], key, grader)

    for side, views in groups.items():
        location = reference_dir if side == "human_regressions" else seed_dir / side
        for view, files in views.items():
            zip_files(location / (view + ".zip"), files)
    releases = {"fi-python-seeds-v1": finish_package(seed_dir, "seeds-card.md"),
                "human-regressions-v1": finish_package(reference_dir, "references-card.md")}
    summary = {s: dict(items=sum(r["split"] == s for r in rows),
                       repositories=len({slug(r) for r in rows if r["split"] == s}),
                       families=len({families[slug(r)] for r in rows if r["split"] == s})) for s in groups}
    if len(references) != len({ref["source"] for ref in references.values()}):
        raise ValueError("Duplicate underlying reference source")
    locked = dict(families={n: dict(facts[n], repo_family_id=families[n]) for n in sorted(facts)},
                  healthy=seeds, references=references, release_sha256=releases)
    lock_file = HERE / "fingerprints.json"
    if freeze:
        lock_file.write_bytes(encode(locked))
        print(f"Frozen {lock_file}")
        return
    if locked != json.loads(lock_file.read_bytes()):
        raise ValueError("Computed source, prompt, fixture, family or release bytes differ from the freeze")
    written = dict(project_commit=git(ROOT, "rev-parse", "HEAD").decode().strip(), host=platform.system(),
                   python=platform.python_version(), counts=summary, family_intersections=overlaps,
                   family_pair_comparisons=comparisons, healthy=healthy_results, references=reference_results,
                   fork_control_counted=False, matched_families=matched, tickets_scanned=len(references),
                   ticket_scan_hits=0, release_sha256=releases)
    (out / "audit.json").write_bytes(encode(written))
    print(json.dumps({k: written[k] for k in ("counts", "family_intersections", "release_sha256")}, indent=2))
    print(f"PASS | audit={out / 'audit.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Optional local bare Git and tokenizer cache, verified on reading")
    parser.add_argument("--output", type=Path, help="A new empty directory for release archives and evidence")
    parser.add_argument("--freeze", action="store_true", help="Maintainer: regenerate the reviewed content fingerprint lock")
    args = parser.parse_args()
    if args.output:
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            parser.error("--output must be empty to keep evidence from separate runs independent")
    else:
        evidence = ROOT / "evidence"
        evidence.mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="stage1-" + sys.platform + "-", dir=evidence))
    with tempfile.TemporaryDirectory(prefix="fi1-") as temporary:
        work = Path(temporary)
        cache = args.cache.resolve() if args.cache else work / "sources"
        cache.mkdir(parents=True, exist_ok=True)
        python, cli, env = prepare_tools(work, ("tokenizers==0.21.4", "jinja2==3.1.6"))
        subprocess.run([str(python), "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_contracts.py"],
                       env=env, check=True)
        subprocess.run([str(python), str(Path(__file__)), "_build", str(work), str(cache), str(out), str(cli),
                        "freeze" if args.freeze else "verify"], env=env, check=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_suite"]:
        sys.exit(run_suite(*sys.argv[2:]))
    elif sys.argv[1:2] == ["_build"]:
        build(*(Path(v) for v in sys.argv[2:6]), freeze=sys.argv[6] == "freeze")
    else:
        main()
