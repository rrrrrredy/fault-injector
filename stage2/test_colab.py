"""CPU integration checks for the Colab handoff. FI_TRAIN_DIR names two prepared, frozen ZIPs."""

import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from stage2.colab_assets import collect_outputs, copy_and_verify
from stage2.colab_grpo import (
    actor_rows,
    make_prompts,
    score_group,
    gpu_inventory,
    check_native_torch,
    metered_worker,
    generation_options,
    MODES,
    MAX_LENGTH,
    COMPLETION,
)
from stage2.state_store import open_state, record_step, snapshot

ROOT = Path(__file__).resolve().parents[1]


class ColabContract(unittest.TestCase):
    def test_notebook(self):
        from nbformat import reads, validate

        text = (ROOT / "notebooks/fi_stage2_gpu_smoke.ipynb").read_text(encoding="utf-8")
        book = reads(text, as_version=4)
        validate(book)
        title = (
            "Fault-Injector-14B · Stage 2 GPU 冒烟与 16K 显存标定（H100 / A100 兼容 · 40GB 预算红线）"
        )
        self.assertEqual(book.metadata.colab.name, title)
        self.assertEqual((book.metadata.accelerator, book.metadata.colab.gpuType), ("GPU", "H100"))
        self.assertNotIn("runtime_version", text)
        self.assertNotIn("2026.10", text)
        self.assertNotIn("userdata.get", text)
        self.assertEqual(book.cells[0].source.splitlines()[0], "#@title " + title)
        self.assertEqual(len(book.cells), 4)
        for index, cell in enumerate(book.cells):
            self.assertEqual(cell.cell_type, "code")
            self.assertFalse(cell.outputs)
            compile(cell.source, f"colab-cell-{index + 1}", "exec")
            self.assertTrue(cell.source.startswith("#@title "))
            self.assertFalse(
                any(
                    isinstance(n, ast.Name) and n.id == "input" for n in ast.walk(ast.parse(cell.source))
                )
            )
        # Run the actual checkout cell against a local release tag, with no Colab or network.
        cell = book.cells[0].source.split("from stage2.colab_assets import prepare")[0]
        with tempfile.TemporaryDirectory() as temp:
            mirror = Path(temp) / "mirror.git"
            subprocess.run(["git", "clone", "--quiet", "--bare", str(ROOT), str(mirror)], check=True)
            ref = "stage2b-gpu-v2"
            subprocess.run(
                ["git", "-C", str(mirror), "update-ref", "refs/tags/" + ref, "HEAD"], check=True
            )
            local = cell.replace(
                "https://github.com/rrrrrredy/fault-injector.git", mirror.as_posix()
            ).replace('dir="/content"', "dir=" + repr(temp))
            subprocess.run([sys.executable, "-c", local], check=True, capture_output=True)
            # Relocating the release tag to different executable content must fail closed.
            empty = (
                subprocess.check_output(["git", "-C", str(mirror), "mktree"], input=b"").decode().strip()
            )
            other = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(mirror),
                    "-c",
                    "user.name=fixture",
                    "-c",
                    "user.email=fixture@example.invalid",
                    "commit-tree",
                    empty,
                    "-m",
                    "alternate release",
                ],
                text=True,
            ).strip()
            subprocess.run(
                ["git", "-C", str(mirror), "update-ref", "refs/tags/" + ref, other], check=True
            )
            bad = subprocess.run([sys.executable, "-c", local], capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("digest", bad.stderr.lower())
        self.assertIn("mock-preflight", book.cells[0].source)
        self.assertIn("smoke-4b", book.cells[1].source)
        self.assertIn("calibrate-14b", book.cells[2].source)
        self.assertIn("drive.mount", book.cells[3].source)
        self.assertNotIn("drive.mount", "".join(c.source for c in book.cells[:3]))
        self.assertNotIn("runtime.unassign", text)

    def test_gpu_gate_and_generation_options(self):
        torch = SimpleNamespace(
            version=SimpleNamespace(cuda="13.0"),
            cuda=SimpleNamespace(
                device_count=lambda: 1,
                is_available=lambda: True,
                is_bf16_supported=lambda *, including_emulation: not including_emulation,
                get_device_properties=lambda i: SimpleNamespace(
                    name="NVIDIA H100 80GB HBM3", total_memory=81559 * 2**20
                ),
            ),
        )
        for name, total in (("NVIDIA H100 80GB HBM3", 81559), ("NVIDIA A100-SXM4-40GB", 40537)):
            dev = gpu_inventory(f"{name}, {total}\n")
            torch.cuda.get_device_properties = lambda i: SimpleNamespace(
                name=name, total_memory=total * 2**20
            )
            check_native_torch(torch, dev)
        for rows in (
            "Tesla T4, 15360",
            "NVIDIA L4, 23034",
            "NVIDIA H100, 20000",
            "NVIDIA H100, 80000\nNVIDIA H100, 80000",
        ):
            with self.assertRaises(RuntimeError):
                gpu_inventory(rows)
        torch.cuda.is_bf16_supported = lambda **kw: False
        with self.assertRaisesRegex(RuntimeError, "bf16"):
            check_native_torch(torch, dev)
        torch.version.cuda = None
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            check_native_torch(torch, dev)
        self.assertEqual(generation_options(False, True)["min_new_tokens"], COMPLETION)
        self.assertEqual(generation_options(True, True)["min_tokens"], COMPLETION)
        self.assertEqual(generation_options(False, False), {})

    def test_whole_device_budget_without_gpu(self):
        device = gpu_inventory("NVIDIA H100 80GB HBM3, 81559")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            report, draft = base / "receipt.json", base / "worker.json"
            start = {"mode": "calibrate-14b", **device}
            worker = [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('{}')",
                str(draft),
            ]

            def probe(sample_expression):
                return [
                    sys.executable,
                    "-u",
                    "-c",
                    "import time; print(1024,flush=True); time.sleep(.15); " + sample_expression,
                ]

            receipt = metered_worker(
                worker,
                probe("print(40960,flush=True); time.sleep(.3); print(40960)"),
                report,
                draft,
                start,
            )
            self.assertEqual(
                (receipt["status"], receipt["whole_gpu_peak_mib"], receipt["within_40gb_budget"]),
                ("completed", 40960, True),
            )
            draft.unlink()
            with self.assertRaisesRegex(RuntimeError, "40961"):
                metered_worker(
                    [sys.executable, "-c", "import time; time.sleep(15)"],
                    probe("print(40961)"),
                    report,
                    draft,
                    start,
                )
            receipt = json.loads(report.read_bytes())
            self.assertEqual(
                (receipt["status"], receipt["gpu_name"], receipt["within_40gb_budget"]),
                ("budget_exceeded", device["gpu_name"], False),
            )
            with self.assertRaises(RuntimeError):
                metered_worker(worker, [sys.executable, "-c", "pass"], report, draft, start)
            self.assertNotEqual(json.loads(report.read_bytes())["status"], "completed")

    def test_actor_only_and_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            actor = Path(tmp) / "actor.zip"
            shutil.copyfile(Path(os.environ["FI_TRAIN_DIR"]) / "actor.zip", actor)
            rows = actor_rows(actor)  # No grader exists anywhere in this directory.
            self.assertEqual(
                [row["seed_id"] for row in rows], ["blinker", "ordered", "termcolor", "time"]
            )
            with ZipFile(actor) as public:
                for row in rows:
                    key = row["seed_id"] + "/"
                    self.assertIn(json.loads(public.read(key + "task.json"))["task"], row["content"])
                    for name in public.namelist():
                        if name.startswith(key) and (name.endswith(".py") or name.endswith("/VERSION")):
                            self.assertIn(public.read(name).decode(), row["content"])
                    self.assertNotIn("oracle_tests/", row["content"])
                    self.assertNotIn("<|im_start|>", row["content"])
            actor.write_bytes(actor.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                actor_rows(actor)

        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self_text = "\n".join(m["content"] for m in messages) + "\nassistant\n"
                return self_text

            def __call__(self, text):
                return {"input_ids": text.split()}

        tiny = [{"seed_id": "time", "content": "Convert minutes to seconds."}]
        rows = make_prompts(Tokenizer(), tiny, calibration=True)
        self.assertEqual(rows[0]["prompt_tokens"] + COMPLETION, MAX_LENGTH)
        self.assertEqual(len(rows), 1)
        with self.assertRaises(ValueError):
            make_prompts(
                Tokenizer(), [{"seed_id": "time", "content": " x" * MAX_LENGTH}], calibration=False
            )

    def test_per_group_rewards_preserve_order_and_shapes(self):
        def scored(pool, sid, text):
            return {"reward": 1.0 if text == "diff" else -1.0, "reason": "fixture"}

        with patch("stage2.colab_grpo.evaluate", side_effect=scored):
            rows = score_group(
                None,
                ["time", "ordered"],
                ["diff", [{"role": "assistant", "content": "bad"}]],
                [[1], [2, 3]],
                [18, 22],
            )
            self.assertEqual([r["result"]["reward"] for r in rows], [1.0, -1.0])
            self.assertEqual([r["completion"] for r in rows], ["diff", "bad"])
            self.assertEqual([r["result"]["completion_tokens"] for r in rows], [1, 2])
            with self.assertRaises(ValueError):
                score_group(None, ["time"], [], [], [])

    def test_packaged_artifacts_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            for mode, step in MODES.items():
                d = work / (mode + "-fixture")
                d.mkdir()
                with open_state(d / "state.sqlite") as con:
                    for i in range(1, step + 1):
                        record_step(
                            con, mode, i, [{"seed": "time", "completion": "x", "result": {"reward": -1}}]
                        )
                    snapshot(con, d / (mode + ".zip"))
                receipt = {
                    "mode": mode,
                    "status": "completed",
                    "project_commit": "f" * 40,
                    "run_dir": d.name,
                    "optimizer_steps": step,
                    "whole_gpu_peak_mib": 40960,
                    "within_40gb_budget": True,
                    "case_status": "no_verified_candidate" if step > 1 else "not_requested",
                }
                (work / (mode + "_receipt.json")).write_text(json.dumps(receipt), encoding="utf-8")
            files = collect_outputs(work)
            target = Path(tmp) / "transport"
            out = copy_and_verify(files, target)
            self.assertEqual({k: v["steps"][k] for k, v in out.items()}, MODES)
            with ZipFile(target / "cases-and-receipts.zip") as z:
                self.assertTrue(all(m + "_receipt.json" in z.namelist() for m in MODES))
            for filename in ("cases-and-receipts.zip", "smoke-4b.zip", "calibrate-14b.zip"):
                self.assertEqual(
                    hashlib.sha256((target / filename).read_bytes()).hexdigest(),
                    (target / (filename + ".sha256")).read_text().split()[0],
                )
            report = work / "calibrate-14b_receipt.json"
            data = json.loads(report.read_bytes())
            data["whole_gpu_peak_mib"] = 40961
            report.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "budget"):
                collect_outputs(work)


if __name__ == "__main__":
    unittest.main()
