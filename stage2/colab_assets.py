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
from urllib.request import urlopen
from zipfile import ZipFile, ZIP_DEFLATED

from .sandbox import TRAIN_HASHES, TRAIN_REVISION, guard_mounts
from .state_store import restore, sha

ROOT = Path(__file__).resolve().parents[1]
NODE = "node-v24.22.3-linux-x64"
NODE_HASH = "4f2b175a6d53a5c617a80dba2676238a0688cef28c6da4eba35c7d844e66fcfb"


def install_colab(work):
    """Resolve in a virtual environment that inherits the image's native CUDA wheels."""
    from packaging.requirements import Requirement

    # A short-lived probe releases its CUDA context before setup and the measured jobs.
    probe = (
        "import json,torch; from stage2.colab_grpo import gpu_inventory,check_native_torch; "
        "d=gpu_inventory(); check_native_torch(torch,d); "
        "print(json.dumps({'torch':str(torch.__version__),'cuda':torch.version.cuda,**d}))"
    )
    native = json.loads(subprocess.check_output([sys.executable, "-c", probe], cwd=ROOT, text=True))
    env = work / "venv"
    # Colab already supplies pip. Inherit it without requiring the image's optional ensurepip bundle.
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", str(env)], check=True
    )
    python = env / "bin/python"
    protected = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in {"torch", "torchvision", "torchaudio", "triton", "cuda-toolkit"} or name.startswith(
            "nvidia-"
        ):
            protected[name] = dist.version
    constraints = work / "native-cuda.txt"
    constraints.write_text("".join(f"{k}=={v}\n" for k, v in protected.items()), encoding="utf-8")
    plan_path = work / "pip-plan.json"
    pip = [str(python), "-m", "pip", "--disable-pip-version-check", "install"]
    resolve = [
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
    ]
    # The project's audited vLLM wheel uses CUDA 13.0 and publishes its exact Torch ABI in PyPI metadata.
    # TRL can also generate directly with the training model, with no second engine or vLLM wheel.
    fast = False
    if native["cuda"] == "13.0":
        with urlopen("https://pypi.org/pypi/vllm/0.22.1/json", timeout=30) as response:
            requires = [Requirement(s) for s in json.load(response)["info"]["requires_dist"]]
        deps = {
            r.name: r.specifier
            for r in requires
            if r.name in {"torch", "torchvision", "torchaudio"} and (not r.marker or r.marker.evaluate())
        }
        fast = "torch" in deps and all(
            n in protected and protected[n] in spec for n, spec in deps.items()
        )
    cuda_tag = "cu" + native["cuda"].replace(".", "")
    if fast:
        # This is the same CUDA xFormers binary used with the audited shared vLLM engine.
        accelerators = [
            "vllm==0.22.1",
            "xformers @ https://download.pytorch.org/whl/cu130/"
            "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            "#sha256=962eb73f7243fb6a6b68ed85ed8f97780070ee35c1be464eefe3299b0382391d",
        ]
    else:
        # xFormers 0.0.36 moved its attention kernels to MSLK. PyPI carries a 0.0.0 placeholder.
        accelerators = [
            "--find-links",
            f"https://download.pytorch.org/whl/{cuda_tag}/mslk/",
            "--only-binary=mslk",
            "mslk>=1",
            "xformers==0.0.36",
        ]
    print(
        f"Native PyTorch {native['torch']}, generation backend: {'vllm' if fast else 'transformers'}",
        flush=True,
    )
    subprocess.run([*resolve, *accelerators], check=True)
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
    if not fast:
        kernel = subprocess.check_output(
            [str(python), "-c", "import importlib.metadata as m; print(m.version('mslk'))"], text=True
        ).strip()
        if not kernel.endswith("+" + cuda_tag):
            raise RuntimeError(f"MSLK wheel {kernel} does not match native PyTorch CUDA {cuda_tag}")
    # Check native extensions inside the resolved environment before opening a training job.
    verify = (
        "from stage2.colab_grpo import check_native_torch,gpu_inventory; import torch; "
        "check_native_torch(torch,gpu_inventory()); from unsloth import FastLanguageModel; "
        "from trl import GRPOTrainer"
    )
    subprocess.run([str(python), "-c", verify], cwd=ROOT, check=True)
    (work / "colab-runtime.json").write_text(
        json.dumps(
            {
                "torch": native["torch"],
                "cuda": native["cuda"],
                "generation_backend": "vllm" if fast else "transformers",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return python


def prepare(work):
    guard_mounts()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        FI_WORK_DIR=str(work), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HOME=str(work / "public-hf")
    )
    python = install_colab(work)
    train = work / "train"
    train.mkdir(exist_ok=True)
    if not all(
        (train / name).is_file() and sha(train / name) == digest for name, digest in TRAIN_HASHES.items()
    ):
        subprocess.run(
            [str(python), "-m", "stage2.check", "prepare-train", "--train", str(train)],
            cwd=ROOT,
            check=True,
        )
    if any(sha(train / name) != digest for name, digest in TRAIN_HASHES.items()):
        raise ValueError(f"Reconstructed train archives differ from frozen HF revision {TRAIN_REVISION}")
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
    from .colab_grpo import GPU_BUDGET_MIB, MODES

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
        if (
            receipt.get("within_40gb_budget") is not True
            or not 0 < receipt["whole_gpu_peak_mib"] <= GPU_BUDGET_MIB
        ):
            raise ValueError(f"{mode} lacks a passing whole-GPU memory budget measurement")
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
