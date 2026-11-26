"""Colab GRPO for four frozen functional slices. The mock CLI only needs Python and Git."""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

from . import sandbox
from .rewards import evaluate, export_case
from .state_store import open_state, record_step, restore, snapshot, cursor_state

MODES = {"smoke-4b": 30, "calibrate-14b": 1}
MODELS = {
    "smoke-4b": ("unsloth/Qwen3-4B-unsloth-bnb-4bit", "bbe12ebf9362b3bcbf44e826e4c98910f2c02e72"),
    "calibrate-14b": ("unsloth/Qwen3-14B-unsloth-bnb-4bit", "46105e245750aad3be7fd1d81c21cb03a0e438ed"),
}
SEEDS = ("blinker", "ordered", "termcolor", "time")
MAX_LENGTH, COMPLETION, GROUP = 16384, 5120, 4
GPU_PACKAGES = ("torch", "unsloth", "vllm", "trl", "datasets", "transformers")
ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "Create one ordinary functional regression for this frozen Python example's stated behavior. "
    "Edit only its existing business source. Preserve imports, tests and their environment. "
    "Return a unified diff with a/ and b/ paths. Optional reasoning ends before the diff."
)


def actor_rows(actor):
    """This boundary has access to the actor file alone, including its published task and public tests."""
    data = Path(actor).read_bytes()
    if hashlib.sha256(data).hexdigest() != sandbox.TRAIN_HASHES["actor.zip"]:
        raise ValueError("Unrecognized train/actor.zip")
    start, end = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
    rows = []
    with ZipFile(BytesIO(data)) as archive:
        for seed in SEEDS:
            rendered = archive.read(seed + "/prompt.txt").decode("utf-8")
            if not rendered.startswith(start) or not rendered.endswith(end):
                raise ValueError(f"Unexpected frozen actor template for {seed}")
            rows.append({"seed_id": seed, "content": rendered[len(start) : -len(end)]})
    return rows


def make_prompts(tokenizer, items, *, calibration):
    # TRL 1.15 tokenizes standard (string) prompts with processing_class(text=prompts).
    def render(item, padding=""):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": padding + item["content"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text)["input_ids"]
        return {"seed_id": item["seed_id"], "prompt": text, "prompt_tokens": len(ids)}

    limit = MAX_LENGTH - COMPLETION
    rows = [render(item) for item in items]
    for row in rows:
        if row["prompt_tokens"] > limit:
            raise ValueError(
                f"{row['seed_id']} exceeds {limit} actual input tokens; refusing to truncate"
            )
    if not calibration:
        return rows
    longest = max(range(len(rows)), key=lambda i: rows[i]["prompt_tokens"])
    item, row = items[longest], rows[longest]
    padding_tokens = limit - row["prompt_tokens"]
    for _ in range(5):
        if row["prompt_tokens"] == limit:
            return [row]
        row = render(
            item,
            "Length calibration only; ignore the following padding:" + " x" * padding_tokens + "\n\n",
        )
        padding_tokens += limit - row["prompt_tokens"]
        if padding_tokens < 0:
            break
    raise ValueError(f"Could not produce the exact calibration input of {limit} tokens")


def score_group(pool, seeds, completions, token_ids, prompt_lengths):
    rows = []
    for seed, completion, ids, length in zip(seeds, completions, token_ids, prompt_lengths, strict=True):
        if isinstance(completion, str):
            text = completion
        elif (
            isinstance(completion, list)
            and len(completion) == 1
            and completion[0].get("role") == "assistant"
            and isinstance(completion[0].get("content"), str)
        ):
            text = completion[0]["content"]
        else:
            raise ValueError("Unexpected TRL completion")
        if length + COMPLETION > MAX_LENGTH or len(ids) > COMPLETION:
            raise ValueError("Rollout exceeded its actual token budget")
        observation = evaluate(pool, seed, text)
        observation.update(prompt_tokens=length, completion_tokens=len(ids))
        rows.append({"seed": seed, "completion": text, "result": observation})
    return rows


def mock_preflight(work=None):
    started = time.perf_counter()
    sandbox.guard_mounts()
    for source in (ROOT / "stage2").glob("*.py"):
        compile(source.read_bytes(), str(source), "exec")
    with tempfile.TemporaryDirectory(prefix="fi-preflight-") as tmp:
        tmp = Path(tmp)
        if work is None:
            # In a fresh offline clone, replace only the asset loader. Git, reward and SQLite remain real.
            from unittest.mock import patch

            files = {"duration.py": b"def seconds(minutes):\n    return minutes * 60\n"}
            fixture = sandbox.Seed("time", files, {}, (), "duration.py", ".", sandbox.fingerprint(files))
            inputs = patch.object(sandbox, "load_train", return_value={"time": fixture})
            train, runtime, source = tmp, tmp, "synthetic-asset"
        else:
            train, runtime, source = work / "train", work / "wasi", "frozen-train"
            actor_rows(train / "actor.zip")
            inputs = nullcontext()
        with inputs, sandbox.Pool(train, runtime=runtime, workers=1) as pool:
            if work is not None:
                # The WASI capability probe needs only empty guest directories. evaluate() checks the worktree.
                probe = sandbox.execute(runtime, pool.root, tmp, pool.seeds["time"], mode="probe")
                if probe.get("network") is not False or probe.get("spawn") is not False:
                    raise RuntimeError(f"Live WASI isolation probe failed: {probe}")
            rows = score_group(pool, ["time"], ["<think>fixture</think>\ninvalid"], [[]], [1])
        obs = rows[0]["result"]
        if (obs["reward"], obs["reason"], obs["reset_matches"]) != (-1.0, "format", True):
            raise RuntimeError(f"Unexpected mock reward: {obs}")
        with open_state(tmp / "run.sqlite") as con:
            record_step(con, "mock", 1, rows, metadata={"next_seed": "time"})
            archive = snapshot(con, tmp / "mock.zip")
        path, checkpoint = restore(archive, tmp / "restored")
        with open_state(path) as con:
            if (
                cursor_state(con)["steps"] != {"mock": 1}
                or checkpoint["rollouts"] != 1
                or checkpoint["metadata"]["mock"] != {"next_seed": "time"}
                or checkpoint["integrity"] != "ok"
            ):
                raise RuntimeError(f"Invalid restored cursor: {checkpoint}")
    loaded = [name for name in GPU_PACKAGES if name in sys.modules]
    if loaded:
        raise RuntimeError(f"Heavy imports before GPU mode: {loaded}")
    return {
        "mode": "mock-preflight",
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        if (ROOT / ".git").exists()
        else None,
        "status": "ok",
        "input": source,
        "reward": obs["reward"],
        "wasi_probe": work is not None,
        "reset_matches": True,
        "restored_step": 1,
        "integrity": "ok",
        "gpu_imports": loaded,
        "elapsed_sec": round(time.perf_counter() - started, 4),
    }


@contextmanager
def gpu_samples(path):
    """Sample device 0 throughout loading and training; fail if the external meter failed."""
    with path.open("w", encoding="ascii") as log:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                "0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=log,
        )
        try:
            yield
            if proc.poll() is not None:
                raise RuntimeError("nvidia-smi stopped during the measurement")
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=10)


def run_gpu(mode, work):
    if not sys.platform.startswith("linux") or not Path("/content").is_dir():
        raise RuntimeError("GPU modes require the user's Colab A100 runtime")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(devices) != 1 or "A100" not in devices[0]:
        raise RuntimeError("Select a single A100 in Colab")
    preflight = mock_preflight(work)
    # A real healthy WASI assertion must pass before any GPU package is imported.
    with sandbox.Pool(work / "train", runtime=work / "wasi", workers=1) as pool:
        if pool.baseline("time")["status"] != "pass":
            raise RuntimeError("Frozen WASI baseline failed")
    from unsloth import FastLanguageModel  # install patches before TRL / transformers
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    work.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=mode + "-", dir=work))
    report = work / (mode + "_receipt.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    model_id, revision = MODELS[mode]
    receipt = {
        "mode": mode,
        "project_commit": commit,
        "run_dir": run.name,
        "status": "running",
        "model": model_id,
        "revision": revision,
        "train_revision": sandbox.TRAIN_REVISION,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "preflight": preflight,
    }
    report.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    weights = snapshot_download(model_id, revision=revision, token=False)
    calibration = mode == "calibrate-14b"
    pending = []
    groups_timed = 0.0
    longest_sequence = 0
    longest_completion = 0
    steps_with_reward_spread = 0
    candidate = None
    monitor = run / "device-mib.txt"
    with (
        open_state(run / "run.sqlite") as con,
        sandbox.Pool(work / "train", runtime=work / "wasi") as pool,
    ):

        def verifier(prompts, completions, completion_ids, seed_id, prompt_tokens, **kwargs):
            nonlocal groups_timed
            before = time.monotonic()
            rows = score_group(pool, seed_id, completions, completion_ids, prompt_tokens)
            pending.extend(rows)
            groups_timed += time.monotonic() - before
            return [r["result"]["reward"] for r in rows]

        class CompletedStep(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                nonlocal candidate, longest_sequence, longest_completion, steps_with_reward_spread
                if len(pending) != GROUP:
                    raise RuntimeError(
                        f"Optimizer step {state.global_step} has {len(pending)} observed rollouts"
                    )
                lengths = [
                    r["result"]["prompt_tokens"] + r["result"]["completion_tokens"] for r in pending
                ]
                if calibration and any(n != MAX_LENGTH for n in lengths):
                    raise RuntimeError(f"16K calibration did not reach its target: {lengths}")
                longest_sequence = max(longest_sequence, *lengths)
                longest_completion = max(
                    longest_completion, *(r["result"]["completion_tokens"] for r in pending)
                )
                steps_with_reward_spread += len({r["result"]["reward"] for r in pending}) > 1
                if candidate is None and not calibration:
                    candidate = next((r for r in pending if r["result"]["reward"] > 0), None)
                record_step(
                    con,
                    mode,
                    state.global_step,
                    pending,
                    metadata={"model": model_id, "revision": revision},
                )
                pending.clear()
                if state.global_step % 10 == 0 and state.global_step < MODES[mode]:
                    snapshot(con, run / (mode + ".zip"))

        torch.cuda.reset_peak_memory_stats()
        with gpu_samples(monitor):
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=weights,
                use_exact_model_name=True,
                max_seq_length=MAX_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                fast_inference=True,
                max_lora_rank=32,
                gpu_memory_utilization=0.58,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=32,
                lora_alpha=32,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                use_gradient_checkpointing="unsloth",
                random_state=3407,
            )
            rows = make_prompts(tokenizer, actor_rows(work / "train/actor.zip"), calibration=calibration)
            cfg = GRPOConfig(
                output_dir=str(run / "trainer"),
                per_device_train_batch_size=1,
                gradient_accumulation_steps=GROUP,
                num_generations=GROUP,
                max_steps=MODES[mode],
                learning_rate=5e-6,
                bf16=True,
                use_vllm=True,
                vllm_mode="colocate",
                vllm_gpu_memory_utilization=0.58,
                vllm_max_model_length=MAX_LENGTH,
                max_completion_length=COMPLETION,
                generation_kwargs={"min_tokens": COMPLETION, "ignore_eos": True} if calibration else {},
                mask_truncated_completions=False,
                report_to="none",
                save_strategy="no",
                logging_steps=1,
            )
            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=[verifier],
                args=cfg,
                train_dataset=Dataset.from_list(rows),
                callbacks=[CompletedStep()],
            )
            if trainer.vllm_generation.llm is not model.vllm_engine:
                raise RuntimeError("TRL did not reuse the Unsloth inference engine")
            torch.cuda.synchronize()
            started = time.monotonic()
            trainer.train()
            torch.cuda.synchronize()
            train_seconds = time.monotonic() - started
        cursor = cursor_state(con)
        if (
            pending
            or cursor["steps"] != {mode: MODES[mode]}
            or trainer.state.global_step != MODES[mode]
            or cursor["rollouts"] != MODES[mode] * GROUP
        ):
            raise RuntimeError(f"Incomplete optimizer steps: {cursor}")
        snapshot(con, run / (mode + ".zip"))
    samples = [int(line) for line in monitor.read_text(encoding="ascii").splitlines() if line.strip()]
    if not samples or max(samples) <= 0:
        raise RuntimeError("No valid whole-device memory samples")
    receipt.update(
        optimizer_steps=trainer.state.global_step,
        rollouts=cursor["rollouts"],
        max_sequence_tokens_observed=longest_sequence,
        max_prompt_tokens=max(r["prompt_tokens"] for r in rows),
        max_completion_tokens=COMPLETION,
        max_completion_tokens_observed=longest_completion,
        steps_with_reward_spread=steps_with_reward_spread,
        whole_gpu_sampled_peak_mib=max(samples),
        device_samples=len(samples),
        device_sample_interval_ms=200,
        torch_max_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
        torch_max_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1),
        train_sec=round(train_seconds, 3),
        sandbox_sec=round(groups_timed, 3),
    )
    receipt["case_status"] = "not_requested" if calibration else "no_verified_candidate"
    if candidate:
        case = export_case(
            work / "train",
            candidate["seed"],
            candidate["completion"],
            runtime=work / "wasi",
            tools=work / "export-tools",
            out=run / "case",
        )
        if case["rci_exit"] != 0 or len(case["cases"]) != 3 or case["source_count"] != 1:
            raise RuntimeError("Invalid Factory/RunCase receipt")
        receipt.update(case_status="verified", rci_exit=0, source_count=1)
    receipt["status"] = "completed"
    report.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock-preflight", *MODES], required=True)
    args = parser.parse_args()
    work = os.getenv("FI_WORK_DIR")
    if args.mode == "mock-preflight":
        result = mock_preflight(Path(work) if work else None)
    else:
        result = run_gpu(args.mode, Path(work or "/content/work_fi"))
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
