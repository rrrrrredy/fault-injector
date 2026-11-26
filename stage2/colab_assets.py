"""Trusted Colab setup and post-run file transfer. Candidates never call this module."""

from hashlib import sha256
from importlib import metadata
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile, ZIP_DEFLATED

from .sandbox import TRAIN_HASHES, TRAIN_REVISION, guard_mounts
from .state_store import restore, sha

ROOT = Path(__file__).resolve().parents[1]
NODE = "node-v24.22.3-linux-x64"
NODE_HASH = "4f2b175a6d53a5c617a80dba2676238a0688cef28c6da4eba35c7d844e66fcfb"


def fetch_inputs(train, get_secret):
    train.mkdir(parents=True, exist_ok=True)
    token = get_secret()
    if not isinstance(token, str) or not token.strip():
        raise ValueError(
            "Allow this notebook to read a Colab secret named HF_TOKEN for the private train dataset"
        )
    for name, expected in TRAIN_HASHES.items():
        url = f"https://huggingface.co/datasets/RedinGhost/fi-python-seeds-v1/resolve/{TRAIN_REVISION}/train/{name}"
        req = Request(url)
        # A signed CDN redirect must not inherit the private Hub Authorization header.
        req.add_unredirected_header("Authorization", "Bearer " + token)
        with urlopen(req, timeout=60) as response:
            data = response.read()
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen HF train/{name} SHA-256 mismatch")
        (train / name).write_bytes(data)


def install_colab(work):
    """Resolve in a virtual environment that inherits the image's native CUDA wheels."""
    if metadata.version("torch") != "2.11.0+cu130" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Select the Colab 2026.10 GPU runtime (Python 3.13, torch 2.11.0+cu130)")
    env = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(env)], check=True)
    python = env / "bin/python"
    protected = {n: metadata.version(n) for n in ("torch", "torchvision", "torchaudio", "triton")}
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name.startswith("nvidia-") or name == "cuda-toolkit":
            protected[name] = dist.version
    constraints = work / "native-cuda.txt"
    constraints.write_text("".join(f"{k}=={v}\n" for k, v in protected.items()), encoding="utf-8")
    plan_path = work / "pip-plan.json"
    pip = [str(python), "-m", "pip", "--disable-pip-version-check", "install"]
    subprocess.run(
        [
            *pip,
            "--dry-run",
            "--report",
            str(plan_path),
            "-c",
            str(constraints),
            "-r",
            str(ROOT / "stage2/requirements-cpu.txt"),
            "-r",
            str(ROOT / "stage2/requirements-colab.txt"),
        ],
        check=True,
    )
    plan = json.loads(plan_path.read_bytes())["install"]
    pins = []
    for item in plan:
        name = item["metadata"]["name"]
        if name.lower().replace("_", "-") in protected:
            raise RuntimeError(f"Resolver proposed replacing native package {name}")
        info = item["download_info"]
        digest = info["archive_info"]["hashes"]["sha256"]
        pins.append(name + " @ " + info["url"].split("#")[0] + "#sha256=" + digest)
    if pins:
        resolved = work / "resolved-colab.txt"
        resolved.write_text("\n".join(pins) + "\n", encoding="utf-8")
        # The complete resolver plan supplies all missing dependencies, with their original digests.
        subprocess.run([*pip, "--no-deps", "-r", str(resolved)], check=True)
    probe = "import importlib.metadata as m,json,sys;print(json.dumps({n:m.version(n) for n in sys.argv[1:]}))"
    actual = json.loads(subprocess.check_output([str(python), "-c", probe, *protected], text=True))
    if actual != protected:
        raise RuntimeError(f"Native CUDA package mismatch: {actual}")
    return python


def prepare(work, get_secret):
    guard_mounts()
    work.mkdir(parents=True, exist_ok=True)
    # No secret is requested during dependency installation or passed to a subprocess.
    os.environ.update(
        FI_WORK_DIR=str(work), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HOME=str(work / "public-hf")
    )
    python = install_colab(work)
    fetch_inputs(work / "train", get_secret)
    node = work / NODE
    if not (node / "bin/npm").is_file():
        with urlopen("https://nodejs.org/dist/v24.22.3/" + NODE + ".tar.xz", timeout=90) as response:
            payload = response.read()
        if sha256(payload).hexdigest() != NODE_HASH:
            raise ValueError("Pinned Node/npm digest mismatch")
        with tarfile.open(fileobj=BytesIO(payload), mode="r:xz") as archive:
            archive.extractall(work, filter="data")
    os.environ["PATH"] = str(node / "bin") + os.pathsep + os.environ["PATH"]
    for module, command, option, dest in (
        ("stage2.sandbox", "prepare", "--runtime", "wasi"),
        ("stage2.rewards", "prepare-export", "--tools", "export-tools"),
    ):
        subprocess.run(
            [str(python), "-m", module, command, option, str(work / dest)], cwd=ROOT, check=True
        )
    return str(python)


def collect_outputs(work):
    """Check both completed jobs and package the Case evidence and receipts before mounting Drive."""
    from .colab_grpo import MODES

    files = []
    archive = work / "cases-and-receipts.zip"
    docs = {}
    commits = set()
    for mode, steps in MODES.items():
        path = work / (mode + "_receipt.json")
        receipt = json.loads(path.read_bytes())
        if (
            receipt["mode"] != mode
            or receipt["status"] != "completed"
            or receipt["optimizer_steps"] != steps
        ):
            raise ValueError(f"Incomplete {mode} receipt")
        commits.add(receipt["project_commit"])
        run = work / receipt["run_dir"]
        if run.resolve().parent != work.resolve():
            raise ValueError("Unexpected local run directory")
        snap = run / (mode + ".zip")
        if sha(snap) != snap.with_suffix(".zip.sha256").read_text().split()[0]:
            raise ValueError(f"Local {mode} snapshot digest mismatch")
        files.append(snap)
        docs[path.name] = path
        if mode == "smoke-4b" and receipt["case_status"] == "verified":
            case = run / "case"
            validated = json.loads((case / "receipt.json").read_bytes())
            if validated["rci_exit"] != 0 or validated["source_count"] != 1:
                raise ValueError("Case package is missing semantic acceptance")
            for filename in (
                "receipt.json",
                "fixture.bundle",
                "candidate.diff",
                *(f"cases/case-{i}.json" for i in range(3)),
            ):
                docs["case/" + filename] = case / filename
    if len(commits) != 1:
        raise ValueError("The two Colab runs used different project commits")
    with ZipFile(archive, "w", ZIP_DEFLATED) as output:
        for name, path in docs.items():
            output.write(path, name)
    archive.with_suffix(".zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n", encoding="ascii"
    )
    return [*files, archive]


def copy_and_verify(files, destination):
    """Copy only archives and sidecars. Restore only from bytes read back into a fresh local directory."""
    from .colab_grpo import MODES

    destination.mkdir(parents=True, exist_ok=True)
    checked = {}
    with tempfile.TemporaryDirectory(prefix="fi-drive-readback-") as temp:
        fresh = Path(temp)
        for archive in files:
            checksum = archive.with_suffix(".zip.sha256")
            expected = checksum.read_text(encoding="ascii").split()[0]
            for part in (archive, checksum):
                remote = destination / part.name
                staged = remote.with_name(remote.name + ".part")
                shutil.copyfile(part, staged)
                staged.replace(remote)
                shutil.copyfile(remote, fresh / part.name)
            if (
                sha(fresh / archive.name) != expected
                or (fresh / checksum.name).read_bytes() != checksum.read_bytes()
            ):
                raise ValueError(f"Transfer checksum mismatch for {archive.name}")
            if archive.stem in MODES:
                _, cursor = restore(fresh / archive.name, fresh / ("restored-" + archive.stem))
                if cursor["steps"] != {archive.stem: MODES[archive.stem]} or cursor["integrity"] != "ok":
                    raise ValueError(f"Readback cursor mismatch: {cursor}")
                checked[archive.stem] = cursor
    return checked
