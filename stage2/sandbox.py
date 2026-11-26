"""Git worktrees on the host, CPython in capability-limited WASI for all candidate execution."""

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
WASM_FILE = "python-3.13.7-wasi_sdk-24.zip"
WASM_URL = "https://github.com/brettcannon/cpython-wasi-build/releases/download/v3.13.7/" + WASM_FILE
WASM_SHA256 = "ea590e5818107bc2abcba01fd98d23f65508d80ed9cdde4e9f15caed5f5f84cb"
WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))
TRAIN_REVISION = "29b9338d1742ec1013bbc72e533eba3c0be8c274"
TRAIN_HASHES = {
    "actor.zip": "a77a82a4535381c76ba5e659f5c63950bf4bd84bdd27501fb3ec31928e0d5a56",
    "grader.zip": "21af5d0297e299e4f16b273efa9c666f3db544113407e83152bbbc6465ee30a5",
}


@dataclass(frozen=True)
class Seed:
    key: str
    files: dict
    checks: dict
    nodes: tuple
    module: str
    import_root: str
    baseline: str


def fingerprint(files):
    hashes = {p: hashlib.sha256(data).hexdigest() for p, data in files.items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def actual_fingerprint(tree):
    return fingerprint(
        {
            p.relative_to(tree).as_posix(): p.read_bytes()
            for p in tree.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(tree).parts
        }
    )


def load_train(train):
    train = Path(train)
    archives = {}
    for name, expected in TRAIN_HASHES.items():
        archives[name] = (train / name).read_bytes()
        if hashlib.sha256(archives[name]).hexdigest() != expected:
            raise ValueError(f"Expected train/{name} from HF commit {TRAIN_REVISION}")
    seeds = {}
    with (
        ZipFile(BytesIO(archives["actor.zip"])) as actors,
        ZipFile(BytesIO(archives["grader.zip"])) as graders,
    ):
        for key in ("blinker", "ordered", "termcolor", "time"):
            prefix = key + "/"
            info = json.loads(graders.read(prefix + "source.json"))
            public, oracle = info["public_node"], info["oracle_node"]
            a, b = public.split("::")[0], oracle.split("::")[0]
            checks = {a: actors.read(prefix + a), b: graders.read(prefix + b)}
            files = {
                p[len(prefix) :]: actors.read(p)
                for p in actors.namelist()
                if p.startswith(prefix)
                and p[len(prefix) :] not in (a, "task.json", "prompt.txt")
                and not p.startswith(prefix + "licenses/")
            }
            seeds[key] = Seed(
                key,
                files,
                checks,
                (public, oracle),
                info["module"],
                info["import_root"],
                fingerprint(files),
            )
    return seeds


def prepare(runtime, archive=None):
    """Trusted, one-time dependency preparation; only read-only guest libraries are exposed."""
    import wasmtime

    runtime = runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = runtime / WASM_FILE
        if not archive.exists():
            try:
                with urlopen(WASM_URL, timeout=45) as response:
                    archive.write_bytes(response.read())
            except (URLError, TimeoutError):
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        "v3.13.7",
                        "--repo",
                        "brettcannon/cpython-wasi-build",
                        "--pattern",
                        WASM_FILE,
                        "--output",
                        str(archive),
                        "--clobber",
                    ],
                    check=True,
                )
    if hashlib.sha256(archive.read_bytes()).hexdigest() != WASM_SHA256:
        raise ValueError("Unrecognized WASI CPython archive")
    with tempfile.TemporaryDirectory(dir=runtime) as tmp:
        fresh = Path(tmp)
        with ZipFile(archive) as z:
            z.extractall(fresh)
        compiled = wasmtime.Module.from_file(wasmtime.Engine(), str(fresh / "python.wasm")).serialize()
        (fresh / "python.cwasm").write_bytes(compiled)
        site = fresh / "site"
        site.mkdir()
        for name in (
            "pytest",
            "_pytest",
            "pluggy",
            "packaging",
            "pygments",
            "iniconfig",
            "py",
            "colorama",
        ):
            spec = importlib.util.find_spec(name)
            if spec is None and name == "colorama":
                continue
            if spec is None:
                raise ModuleNotFoundError(name)
            src = Path(spec.origin)
            if src.name == "__init__.py":
                shutil.copytree(src.parent, site / name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copyfile(src, site / src.name)
        (fresh / "support").mkdir()
        shutil.copyfile(HERE / "_guest.py", fresh / "support" / "guest.py")
        for name in ("lib", "site", "support", "python.cwasm"):
            target = runtime / name
            if target.is_dir():
                shutil.copytree(fresh / name, target, dirs_exist_ok=True)
            else:
                (fresh / name).replace(target)
    print(f"Prepared WASI CPython 3.13.7 in {runtime}", flush=True)


def drive_roots(extra=()):
    paths = [Path("/content/drive"), Path("/content/gdrive"), *map(Path, extra)]
    if os.name == "nt":
        for root in os.listdrives():
            paths.extend(Path(root) / name for name in ("My Drive", "我的云端硬盘"))
    mounts = Path("/proc/self/mountinfo")
    if mounts.is_file():
        for line in mounts.read_text(encoding="utf-8", errors="replace").splitlines():
            left, _, right = line.partition(" - ")
            if any(
                name in right.lower() for name in ("drivefs", "google-drive", "google_drive", "gdrive")
            ):
                paths.append(Path(left.split()[4].replace("\\040", " ")))
    return paths


def guard_mounts(extra=(), *, exempt=()):
    exempt = {Path(p).resolve() for p in exempt}
    for path in drive_roots(extra):
        if path.is_dir() and os.access(path, os.R_OK) and path.resolve() not in exempt:
            raise PermissionError(f"Readable Drive path before candidate execution: {path}")


def git(work, *args, data=None):
    binary = shutil.which("git")
    if not binary:
        raise FileNotFoundError("git")
    argv = [binary, "-c", "core.autocrlf=false", "-c", "core.safecrlf=false", "-C", str(work), *args]
    rc, stdout, stderr = supervised(argv, work, 20, input_data=data)
    if rc:
        raise subprocess.CalledProcessError(rc, argv, stdout, stderr)
    return stdout.encode()


def commit(tree, message):
    git(tree, "add", "--force", "--", ".")
    git(
        tree,
        "-c",
        "user.name=Fault Injector",
        "-c",
        "user.email=case@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return git(tree, "rev-parse", "HEAD").decode().strip()


def write_files(root, files):
    for name, data in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


class Pool:
    def __init__(
        self, train, *, runtime, workers=WORKERS, timeout=12, drive_paths=(), exempt_drive_paths=()
    ):
        self.drive_paths = tuple(drive_paths)
        self.exempt_drive_paths = tuple(exempt_drive_paths)
        guard_mounts(self.drive_paths, exempt=self.exempt_drive_paths)
        if workers < 1 or timeout <= 0:
            raise ValueError("workers and timeout must be positive")
        self.seeds = load_train(train)
        self.timeout = timeout
        self.runtime = Path(runtime).resolve()
        parent = Path("/dev/shm") if sys.platform.startswith("linux") else None
        self.temp = tempfile.TemporaryDirectory(prefix="fi-candidate-", dir=parent)
        self.root = Path(self.temp.name).resolve()
        self.queue = queue.Queue()
        for i in range(workers):
            self.queue.put(i)
        self.lock = threading.Lock()
        self.lifecycle = threading.Condition()
        self.active = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        with self.lifecycle:
            self.closed = True
            self.lifecycle.wait_for(lambda: self.active == 0)
        self.temp.cleanup()

    @contextmanager
    def checkout(self, seed, observation=None):
        guard_mounts(self.drive_paths, exempt=self.exempt_drive_paths)
        with self.lifecycle:
            if self.closed:
                raise RuntimeError("Candidate worktree pool is closed")
            self.active += 1
        index = self.queue.get()
        try:
            repo = self.root / (seed.key + "-repo")
            checks = self.root / (seed.key + "-checks")
            tree = self.root / (seed.key + "-" + str(index))
            with self.lock:
                if not repo.exists():
                    write_files(repo, seed.files)
                    write_files(checks, seed.checks)
                    git(repo, "init", "--quiet")
                    commit(repo, "Frozen functional slice")
                if not tree.exists():
                    git(repo, "worktree", "add", "--detach", "--quiet", str(tree), "HEAD")
            try:
                if actual_fingerprint(tree) != seed.baseline:
                    raise RuntimeError("Dirty worktree before evaluation")
                yield tree, checks
            finally:
                git(tree, "reset", "--hard", "--quiet", "HEAD")
                git(tree, "clean", "-ffdqx")
                reset = actual_fingerprint(tree)
                if reset != seed.baseline:
                    raise RuntimeError("Worktree reset did not restore the frozen bytes")
                if observation is not None:
                    observation.update(
                        baseline_fingerprint=seed.baseline, reset_fingerprint=reset, reset_matches=True
                    )
        finally:
            self.queue.put(index)
            with self.lifecycle:
                self.active -= 1
                self.lifecycle.notify_all()

    def verify(self, seed_id, diff, *, timeout=None):
        """Evaluate one UTF-8 unified diff against the two frozen business tests."""
        seed = self.seeds[seed_id]
        timeout = self.timeout if timeout is None else timeout
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        start = time.perf_counter()
        result = dict(
            seed=seed_id,
            reason="format",
            files_changed=0,
            paths=[],
            failed_nodes=[],
            applied=False,
            syntax_ok=False,
            import_ok=False,
            reverse_passed=False,
            faulty_exit_code=None,
            fixed_exit_code=None,
        )
        with self.checkout(seed, result) as (tree, checks):
            try:
                _verify(tree, checks, self.runtime, seed, diff, timeout, result)
            except subprocess.TimeoutExpired:
                result["reason"] = "timeout"
        result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    def baseline(self, seed_id):
        seed = self.seeds[seed_id]
        with self.checkout(seed) as (tree, checks):
            return execute(self.runtime, tree, checks, seed, timeout=self.timeout)


def _verify(tree, checks, runtime, seed, diff, timeout, result):
    if not isinstance(diff, str) or not diff or len(diff.encode()) > 128 * 1024:
        return
    if any(
        line.startswith(
            (
                "new file mode",
                "deleted file mode",
                "old mode",
                "new mode",
                "rename from",
                "copy from",
                "GIT binary patch",
            )
        )
        for line in diff.splitlines()
    ):
        result["reason"] = "path"
        return
    data = diff.encode()
    try:
        stat = git(tree, "apply", "--numstat", "-z", "-", data=data).decode()
        paths = sorted({line.split("\t", 2)[2] for line in stat.split("\0") if line})
    except (subprocess.CalledProcessError, IndexError):
        return
    if not paths:
        return
    result.update(paths=paths, files_changed=len(paths))
    if any(p not in seed.files or not p.endswith(".py") for p in paths):
        result["reason"] = "path"
        return
    try:
        git(tree, "apply", "--index", "--whitespace=nowarn", "-", data=data)
    except subprocess.CalledProcessError:
        result["reason"] = "apply"
        return
    result["applied"] = True
    # Syntax checks run in the guest; the host never imports or compiles submitted Python.
    changed = execute(runtime, tree, checks, seed, timeout=timeout, extra=paths)
    result["faulty_exit_code"] = changed.get("pytest_exit")
    result["syntax_ok"] = changed.get("status") not in ("syntax", "guest_error")
    result["import_ok"] = changed.get("status") in ("pass", "assertion", "test_error")
    result["reason"] = changed["status"]
    if "diagnostic" in changed:
        result["diagnostic"] = changed["diagnostic"]
    if changed["status"] != "assertion":
        return
    result["failed_nodes"] = changed["failed_nodes"]
    git(tree, "apply", "--index", "--reverse", "--whitespace=nowarn", "-", data=data)
    if actual_fingerprint(tree) != seed.baseline:
        raise RuntimeError("Inverse patch differs from the original frozen source")
    fixed = execute(runtime, tree, checks, seed, timeout=timeout, extra=paths)
    result["fixed_exit_code"] = fixed.get("pytest_exit")
    result["reverse_passed"] = fixed["status"] == "pass"
    result["reason"] = "verified" if result["reverse_passed"] else "reverse_failed"


def supervised(argv, cwd, timeout, *, input_data=None):
    posix = os.name == "posix"
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "HOME": str(cwd),
        "TMPDIR": str(cwd),
        "TEMP": str(cwd),
        "TMP": str(cwd),
        "LC_ALL": "C",
        "PYTHONUTF8": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if not posix:
        env["SystemRoot"] = os.environ["SystemRoot"]
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        start_new_session=posix,
        stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    try:
        stdout, stderr = proc.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        if posix:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            taskkill = Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"
            subprocess.run(
                [str(taskkill), "/PID", str(proc.pid), "/T", "/F"],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if proc.poll() is None:
                proc.kill()
        proc.communicate()
        if sys.platform.startswith("linux"):
            while True:
                try:
                    os.waitpid(-proc.pid, 0)
                except ChildProcessError:
                    break
        raise
    return proc.returncode, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


def guest_worker(runtime, tree, checks, tmp, root, nodes, mode, hash_seed, extra):
    import wasmtime as wasm

    engine = wasm.Engine()
    module = wasm.Module.deserialize_file(engine, str(runtime / "python.cwasm"))
    store = wasm.Store(engine)
    store.set_limits(memory_size=256 << 20)
    config = wasm.WasiConfig()
    config.argv = ["python", "-B", "/support/guest.py", mode, root, nodes, *extra]
    config.env = [
        ("HOME", "/tmp"),
        ("TMPDIR", "/tmp"),
        ("PATH", "/unavailable"),
        ("LC_ALL", "C"),
        ("PYTHONHOME", "/runtime"),
        ("PYTHONPATH", "/site"),
        ("PYTHONHASHSEED", hash_seed),
        ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1"),
        ("PYTHONIOENCODING", "utf-8"),
    ]
    for host, path in [
        (runtime / "lib", "/runtime/lib"),
        (runtime / "site", "/site"),
        (runtime / "support", "/support"),
        (tree, "/work"),
        (checks, "/checks"),
    ]:
        config.preopen_dir(str(host), path, fs_mutable=False)
    config.preopen_dir(str(tmp), "/tmp", fs_mutable=False)
    output = bytearray()

    def capture(data):
        output.extend(data[: max(0, 65536 - len(output))])
        return len(data)

    config.stdout_custom = capture
    config.stderr_custom = capture
    store.set_wasi(config)
    linker = wasm.Linker(engine)
    linker.define_wasi()
    instance = linker.instantiate(store, module)
    try:
        instance.exports(store)["_start"](store)
    except wasm.ExitTrap as error:
        if error.code:
            sys.stdout.buffer.write(output)
            return error.code
    sys.stdout.buffer.write(output)
    return 0


def execute(
    runtime, tree, checks, seed, mode="pytest", timeout=12, extra=(), hash_seed="1729", nodes=None
):
    nodes = list(seed.nodes if nodes is None else nodes)
    with tempfile.TemporaryDirectory(prefix="guest-", dir=tree.parent) as tmp:
        module = (
            Path(seed.module)
            .relative_to(seed.import_root)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
            .removesuffix(".__init__")
        )
        argv = [
            sys.executable,
            "-I",
            "-B",
            str(HERE / "sandbox.py"),
            "_guest",
            str(runtime),
            str(tree),
            str(checks),
            tmp,
            seed.import_root,
            json.dumps(nodes),
            mode,
            hash_seed,
            module,
            *extra,
        ]
        rc, stdout, stderr = supervised(argv, Path(tmp), timeout)
    if rc:
        return {"status": "guest_error", "failed_nodes": [], "diagnostic": (stdout + stderr)[-3500:]}
    results = [
        line.removeprefix("FI_GUEST_RESULT ")
        for line in stdout.splitlines()
        if line.startswith("FI_GUEST_RESULT ")
    ]
    if len(results) != 1:
        return {"status": "guest_error", "failed_nodes": [], "diagnostic": (stdout + stderr)[-3500:]}
    try:
        result = json.loads(results[0])
    except (ValueError, RecursionError):
        return {"status": "guest_error", "failed_nodes": [], "diagnostic": "Invalid guest result"}
    if not isinstance(result, dict):
        return {"status": "guest_error", "failed_nodes": [], "diagnostic": "Invalid guest record"}
    if mode == "pytest" and (
        result.get("status") not in ("syntax", "import", "assertion", "pass", "test_error")
        or not isinstance(result.get("failed_nodes"), list)
        or any(n not in nodes for n in result["failed_nodes"])
    ):
        return {
            "status": "guest_error",
            "failed_nodes": [],
            "diagnostic": "Unexpected guest observation",
        }
    if result.get("status") == "test_error":
        result["diagnostic"] = stdout[-2500:]
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["_guest"]:
        sys.exit(guest_worker(*(Path(v) for v in sys.argv[2:6]), *sys.argv[6:10], sys.argv[10:]))
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    prepare(args.runtime, args.archive)
