"""Trusted verifier inside WASI CPython. No host filesystem or environment is inherited."""

import json
import os
from pathlib import Path
import importlib
import sys


def main():
    mode, source_root, nodes, module, *extra = sys.argv[1:]
    os.chdir("/work")
    sys.path.insert(0, "/work/" + source_root)
    nodes = json.loads(nodes)
    if mode == "probe":
        try:
            import socket

            connection = socket.socket()
        except (AttributeError, ImportError, OSError):
            network = False
        else:
            connection.close()
            network = True
        try:
            import subprocess

            subprocess.run(["python", "-V"], check=True)
        except (OSError, ImportError, NotImplementedError):
            spawn = False
        else:
            spawn = True
        return {
            "env_keys": sorted(os.environ),
            "private_reads": [os.path.exists(p) and os.access(p, os.R_OK) for p in extra],
            "network": network,
            "spawn": spawn,
        }
    if mode != "pytest":
        raise ValueError(mode)
    for name in extra:
        try:
            compile((Path("/work") / name).read_bytes(), name, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, OverflowError) as exc:
            return {"status": "syntax", "failed_nodes": [], "diagnostic": str(exc)}
    try:
        importlib.import_module(module)
    except Exception as exc:
        return {
            "status": "import",
            "failed_nodes": [],
            "diagnostic": type(exc).__name__ + ": " + str(exc),
        }

    import pytest

    passed, failed, errors = [], [], []

    class Observations:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_makereport(self, item, call):
            report = (yield).get_result()
            if report.when == "call" and report.passed:
                passed.append(item.nodeid)
            elif report.failed:
                origin = str(call.excinfo.traceback[-1].path) if call.excinfo else ""
                if (
                    call.when == "call"
                    and call.excinfo
                    and call.excinfo.type is AssertionError
                    and origin.startswith("/checks/")
                ):
                    failed.append(item.nodeid)
                else:
                    errors.append(item.nodeid)

    targets = ["/checks/" + node for node in nodes]
    rc = int(
        pytest.main(
            [
                "-q",
                "--capture=sys",
                "--rootdir=/checks",
                "--tb=short",
                "--color=no",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:faulthandler",
                "-p",
                "no:logging",
                *targets,
            ],
            plugins=[Observations()],
        )
    )
    complete = set(passed + failed) == set(nodes)
    if rc not in (0, 1) or errors or not complete:
        return {"status": "test_error", "failed_nodes": [], "pytest_exit": rc}
    return {
        "status": "assertion" if failed else "pass",
        "failed_nodes": sorted(failed),
        "pytest_exit": rc,
    }


print(
    "FI_GUEST_RESULT "
    + json.dumps(dict(main(), hash_seed=os.environ["PYTHONHASHSEED"], hash_probe=hash("fault-injector")))
)
