"""Offline Stage 2 acceptance. The prepare command only reconstructs the four frozen training inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import difflib
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
from urllib.request import urlopen
from zipfile import ZipFile

from .sandbox import Pool, guard_mounts, supervised, execute
from .rewards import evaluate, export_case, reward
from .state_store import open_state, record_step, snapshot, restore, cursor_state

ROOT = Path(__file__).resolve().parents[1]


def prepare_train(destination):
    sys.path.insert(0, str(ROOT / "stage1"))
    from build import selected_test, zip_files
    from corpus import json_bytes, slug

    rows = json.loads((ROOT / "stage1/manifest.json").read_bytes())["repositories"]
    lock = json.loads((ROOT / "stage1/fingerprints.json").read_bytes())
    actors, graders = {}, {}
    for row in (r for r in rows if r["split"] == "train"):
        url = f"https://codeload.github.com/{slug(row)}/zip/{row['commit_sha']}"
        with urlopen(url, timeout=60) as response, ZipFile(BytesIO(response.read())) as upstream:
            key = row["slice_id"]
            prefix = upstream.namelist()[0].split("/")[0] + "/"
            files = {p: upstream.read(prefix + p) for p in row["files"]}
            public, data = selected_test(row, "public")
            files[public] = data
            content = (
                row["task"]
                + "\n"
                + "".join(f"\n--- {n} ---\n{v.decode('utf-8')}" for n, v in sorted(files.items()))
            )
            files["prompt.txt"] = (
                "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n"
            ).encode()
            files["task.json"] = json_bytes({"task": row["task"]})
            files["licenses/" + row["notice"]] = upstream.read(prefix + row["notice"])
            actors.update({key + "/" + n: v for n, v in files.items()})
            info = {
                n: row[n]
                for n in ("repository", "commit_sha", "spdx", "notice", "module", "import_root")
            }
            info.update(lock["families"][slug(row)])
            info.update({n: v for n, v in lock["healthy"][key].items() if n != "split"})
            node, test = selected_test(row, "oracle")
            graders[key + "/source.json"] = json_bytes(info)
            graders[key + "/" + node] = test
    zip_files(destination / "actor.zip", actors)
    zip_files(destination / "grader.zip", graders)


def difference(seed, old, new):
    path = seed.module
    before = seed.files[path].decode()
    if before.count(old) != 1:
        raise ValueError(f"Fixture no longer matches: {old}")
    after = before.replace(old, new)
    return "".join(
        difflib.unified_diff(
            before.splitlines(True), after.splitlines(True), fromfile="a/" + path, tofile="b/" + path
        )
    )


def require(value, message):
    if not value:
        raise AssertionError(message)


def check(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use a fresh evidence directory")
    receipt = {
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "platform": sys.platform,
        "host_python": sys.version.split()[0],
        "wasmtime": version("wasmtime"),
    }
    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch).resolve()
        private = temp / "Drive"
        private.mkdir()
        marker = private / "private-marker.txt"
        marker.write_text("synthetic canary")
        try:
            guard_mounts([private])
        except PermissionError:
            receipt["readable_drive_blocked"] = True
        else:
            raise AssertionError("Accessible Drive path accepted")
        guard_mounts([private], exempt=[private])
        with patch("stage2.state_store.drive_roots", return_value=[private]):
            try:
                with open_state(private / "active.sqlite"):
                    raise AssertionError("Active SQLite database opened on a Drive path")
            except PermissionError:
                require(not (private / "active.sqlite").exists(), private)
                receipt["drive_database_blocked"] = True
        with patch.dict(
            os.environ,
            {
                k: "synthetic-host-canary"
                for k in ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS")
            },
        ):
            with Pool(args.train, runtime=args.runtime, workers=2, timeout=15) as pool:
                scratch_pool = pool.root
                seed = pool.seeds["time"]
                with pool.checkout(seed) as (tree, checks):
                    view = execute(
                        pool.runtime, tree, checks, seed, "probe", extra=[str(private), str(marker)]
                    )
                require(
                    view["private_reads"] == [False, False]
                    and not view["network"]
                    and not view["spawn"],
                    view,
                )
                require(
                    set(view["env_keys"])
                    <= {
                        "HOME",
                        "TMPDIR",
                        "PATH",
                        "LC_ALL",
                        "PYTHONHASHSEED",
                        "PYTHONPATH",
                        "PYTHONHOME",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
                        "PYTHONIOENCODING",
                        "PYTHONUTF8",
                        "PYTHONDONTWRITEBYTECODE",
                        "PYTHON_COLORS",
                        "PYTEST_ADDOPTS",
                    },
                    view,
                )
                receipt["guest_environment"] = view
                with ThreadPoolExecutor(2) as ex:
                    greens = dict(zip(pool.seeds, ex.map(pool.baseline, pool.seeds)))
                require(all(r["status"] == "pass" for r in greens.values()), greens)
                receipt["frozen_train_baselines"] = greens
                line = '        ("mins", 60),'
                good = difference(seed, line, line.replace("60", "61"))
                anchor = "    for timefmt in COMPILED_TIMEFORMATS:"
                loop = difference(seed, anchor, "    while True:\n        pass\n" + anchor)
                candidates = {
                    "malformed": "broken input",
                    "valid": "<think>synthetic test</think>\n" + good,
                    "syntax": difference(seed, line, line.replace("60", "(")),
                    "import": difference(
                        seed, "import re\n", "import unavailable_fixture_module\nimport re\n"
                    ),
                }
                scored = {k: evaluate(pool, "time", v) for k, v in candidates.items()}
                scored["timeout"] = evaluate(pool, "time", loop, timeout=1.5)
                for k, why in (
                    ("malformed", "format"),
                    ("valid", "verified"),
                    ("timeout", "timeout"),
                    ("syntax", "syntax"),
                    ("import", "import"),
                ):
                    require(
                        scored[k]["reason"] == why and scored[k]["reward"] == reward(scored[k]),
                        (k, scored[k]),
                    )
                require(
                    scored["valid"]["reward"] > 0 and scored["valid"]["failed_nodes"], scored["valid"]
                )
                require(all(r["reward"] <= 0 for k, r in scored.items() if k != "valid"), scored)
                require(all(r["reset_matches"] for r in scored.values()), scored)
                receipt["scores"] = scored
                receipt["pool_parent"] = str(pool.root.parent)
            require(not scratch_pool.exists(), scratch_pool)
            receipt["pool_removed"] = True
        # Native supervision uses harmless sleepers; WASI itself cannot start host processes.
        pid_file = temp / "tree.json"
        child = "import subprocess,sys,time,os,json;from pathlib import Path;p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);Path(sys.argv[1]).write_text(json.dumps([os.getpid(),p.pid]));time.sleep(60)"
        parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]);time.sleep(60)"
        try:
            supervised([sys.executable, "-c", parent, child, str(pid_file)], temp, 2)
        except subprocess.TimeoutExpired:
            pids = json.loads(pid_file.read_text())
            if os.name == "posix":
                require(Path(receipt["pool_parent"]) == Path("/dev/shm"), receipt)
                require(all(not Path(f"/proc/{p}").exists() for p in pids), pids)
            else:
                import ctypes

                for pid in pids:
                    handle = ctypes.windll.kernel32.OpenProcess(0x100000, False, pid)
                    if handle:
                        try:
                            require(ctypes.windll.kernel32.WaitForSingleObject(handle, 3000) == 0, pid)
                        finally:
                            ctypes.windll.kernel32.CloseHandle(handle)
            receipt["process_tree"] = {
                "descendant_pids": pids,
                "survivors": 0,
                "method": "killpg" if os.name == "posix" else "taskkill /T /F",
            }
        else:
            raise AssertionError("Expected process-group timeout")
        with open_state(out / "run.sqlite") as con:
            require(con.execute("PRAGMA journal_mode").fetchone()[0] == "wal", "WAL not enabled")
            record_step(
                con,
                "cpu",
                1,
                [{"seed": "time", "completion": good, "result": scored["valid"]}],
                metadata={"next_seed": "blinker"},
            )
            arc = snapshot(con, out / "cpu-snapshot.zip")
            record_step(
                con,
                "cpu",
                2,
                [{"seed": "time", "completion": "broken input", "result": scored["malformed"]}],
            )
        target, summary = restore(arc, temp / "restored")
        require(
            summary["steps"] == {"cpu": 1}
            and summary["rollouts"] == 1
            and summary["integrity"] == "ok"
            and summary["metadata"]["cpu"] == {"next_seed": "blinker"},
            summary,
        )
        with open_state(target) as con:
            record_step(
                con, "cpu", 2, [{"seed": "time", "completion": loop, "result": scored["timeout"]}]
            )
            resumed = cursor_state(con)
        require(resumed["steps"] == {"cpu": 2} and resumed["rollouts"] == 2, resumed)
        receipt["snapshot"] = dict(
            summary, resumed=resumed, archive_sha256=hashlib.sha256(arc.read_bytes()).hexdigest()
        )
        receipt["export"] = export_case(
            args.train, "time", good, runtime=args.runtime, tools=args.tools, out=out / "export"
        )
    receipt["gpu_imports"] = sorted({"torch", "vllm", "unsloth", "trl"} & sys.modules.keys())
    require(not receipt["gpu_imports"], receipt["gpu_imports"])
    (out / "cpu-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("pool_parent", "pool_removed", "process_tree", "snapshot", "gpu_imports")
            },
            indent=2,
        )
    )
    print("PASS", out / "cpu-receipt.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare-train", "verify"])
    p.add_argument("--train", type=Path, required=True)
    for opt in ("runtime", "tools", "out"):
        p.add_argument("--" + opt, type=Path)
    args = p.parse_args()
    if args.command == "prepare-train":
        prepare_train(args.train)
    else:
        if not all((args.runtime, args.tools, args.out)):
            p.error("verify also needs --runtime, --tools and --out")
        check(args)
