"""Fetch the pinned inputs, build the verifier tools, and replay in a fresh directory."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent


def fetch_tool(work, name, revision):
    url = f"https://codeload.github.com/rrrrrredy/{name}/zip/{revision}"
    with urlopen(url, timeout=90) as response, ZipFile(BytesIO(response.read())) as archive:
        archive.extractall(work)
    return work / f"{name}-{revision}"


def checkout_regression(path, url, revision):
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "fetch", "--quiet", "--no-tags", "--depth=2", url, revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "-c", "core.autocrlf=false", "checkout", "--quiet", "--detach", revision],
        check=True,
    )


def replay(work, evidence):
    lock = json.loads((ROOT / "stage0.lock.json").read_text(encoding="utf-8"))
    protocol = fetch_tool(work, "runcase-interchange", lock["upstreams"]["runcase-interchange"])
    factory = fetch_tool(work, "workflow-environment-factory", lock["upstreams"]["workflow-environment-factory"])
    for name, expected in lock["schema_sha256"].items():
        for directory in (protocol / "schemas", ROOT / "schemas"):
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Schema SHA-256 mismatch: {directory / name}")

    source = lock["source"]
    checkout_regression(work / "humanize", source["repository"], source["fix"])
    parent = subprocess.check_output(
        ["git", "-C", str(work / "humanize"), "rev-parse", "HEAD^"], text=True
    ).strip()
    if parent != source["fault"]:
        raise ValueError(f"Expected the pinned faulty parent, got {parent}")

    env_dir = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
    windows = os.name == "nt"
    python = env_dir / ("Scripts/python.exe" if windows else "bin/python")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
         "--constraint", str(factory / "requirements.lock"), "jsonschema", "pydantic", "pytest"],
        check=True,
    )
    npm = shutil.which("npm.cmd" if windows else "npm")
    if npm is None:
        raise SystemExit("Install Node.js with npm and put it on PATH.")
    subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=protocol, check=True)
    cli_prefix = work / "cli"
    subprocess.run(
        [npm, "install", "--global", "--prefix", str(cli_prefix), "--offline", "--install-links=false",
         "--ignore-scripts", "--no-audit", "--no-fund", str(protocol)],
        cwd=protocol, check=True,
    )
    cli = cli_prefix / ("rci.cmd" if windows else "bin/rci")

    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": os.pathsep.join([str(factory / "backend"), str(ROOT / "fixtures")]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "PYTHONOPTIMIZE": "",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    })
    subprocess.run(
        [str(python), str(ROOT / "materialize.py"), str(work), str(evidence), str(cli)],
        env=environment, check=True,
    )


if __name__ == "__main__":
    evidence_root = ROOT / "evidence"
    evidence_root.mkdir(exist_ok=True)
    evidence_dir = Path(tempfile.mkdtemp(prefix=sys.platform + "-", dir=evidence_root))
    print(f"Evidence directory: {evidence_dir}", flush=True)
    with tempfile.TemporaryDirectory(prefix="fi0-") as temp_dir:
        replay(Path(temp_dir), evidence_dir)
