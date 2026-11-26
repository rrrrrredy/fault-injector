"""CPU integration checks for the Colab handoff. FI_TRAIN_DIR names two prepared, frozen ZIPs."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from stage2.colab_assets import collect_outputs, copy_and_verify
from stage2.colab_grpo import actor_rows, make_prompts, score_group, MODES, MAX_LENGTH, COMPLETION
from stage2.state_store import open_state, record_step, snapshot

ROOT = Path(__file__).resolve().parents[1]


class ColabContract(unittest.TestCase):
    def test_notebook(self):
        from nbformat import reads, validate

        text = (ROOT / "notebooks/colab_launcher.ipynb").read_text(encoding="utf-8")
        book = reads(text, as_version=4)
        validate(book)
        self.assertEqual(len(book.cells), 4)
        for index, cell in enumerate(book.cells):
            self.assertEqual(cell.cell_type, "code")
            self.assertFalse(cell.outputs)
            compile(cell.source, f"colab-cell-{index + 1}", "exec")
        self.assertIn("mock-preflight", book.cells[0].source)
        self.assertIn("smoke-4b", book.cells[1].source)
        self.assertIn("calibrate-14b", book.cells[2].source)
        self.assertIn("drive.mount", book.cells[3].source)
        self.assertNotIn("drive.mount", "".join(c.source for c in book.cells[:3]))
        self.assertNotIn("runtime.unassign", text)

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


if __name__ == "__main__":
    unittest.main()
