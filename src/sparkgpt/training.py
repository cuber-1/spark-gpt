"""Single-GPU pretraining loop with atomic, exact-resume checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sparkgpt.config import TrainConfig
from sparkgpt.data import TokenBatcher, file_sha256
from sparkgpt.model import GPT


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, config: TrainConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_lr
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    ratio = (step - config.warmup_steps) / decay_steps
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_lr + coefficient * (config.learning_rate - config.min_lr)


def _git_state() -> dict[str, Any] | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        )
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repository, stderr=subprocess.DEVNULL
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
        ).split(b"\0")
        digest = hashlib.sha256()
        digest.update(sha.encode())
        digest.update(diff)
        for raw_path in sorted(path for path in untracked if path):
            path = repository / os.fsdecode(raw_path)
            digest.update(raw_path)
            if path.is_file():
                digest.update(path.read_bytes())
        return {"sha": sha, "dirty": bool(status), "tree_sha256": digest.hexdigest()}
    except (OSError, subprocess.CalledProcessError):
        return None


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    step: int,
    best_val_loss: float,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
    data_manifest: dict[str, Any],
    initial_git_state: dict[str, Any] | None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    checkpoint = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "rng": _rng_state(),
        "train_batcher": train_batcher.state_dict(),
        "val_batcher": val_batcher.state_dict(),
        "data_manifest": data_manifest,
        "initial_git_state": initial_git_state,
        "git_state": _git_state(),
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_batcher: TokenBatcher,
    val_batcher: TokenBatcher,
    config: TrainConfig,
    data_manifest: dict[str, Any],
) -> tuple[int, float, dict[str, Any] | None]:
    # Sampler permutations and RNG state are always CPU tensors. Model and optimizer
    # loaders copy their tensors to the already-created parameter devices.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    if checkpoint["config"] != config.to_dict():
        raise ValueError("checkpoint configuration does not exactly match the requested run")
    if checkpoint["data_manifest"] != data_manifest:
        raise ValueError("checkpoint data manifest does not match the current token files")
    current_git_state = _git_state()
    initial_git_state = checkpoint.get("initial_git_state") or checkpoint.get("git_state")
    if initial_git_state is not None and current_git_state != initial_git_state:
        raise ValueError("exact resume requires the same Git commit and working-tree fingerprint")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    train_batcher.load_state_dict(checkpoint["train_batcher"])
    val_batcher.load_state_dict(checkpoint["val_batcher"])
    _restore_rng(checkpoint["rng"])
    return int(checkpoint["step"]), float(checkpoint["best_val_loss"]), initial_git_state


@torch.inference_mode()
def evaluate(
    model: GPT,
    batcher: TokenBatcher,
    batches: int,
    autocast_context: Any,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = batcher.next()
        with autocast_context():
            _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.detach().float())
    model.train()
    result = torch.stack(losses).mean().item()
    if not math.isfinite(result):
        raise FloatingPointError("non-finite validation loss; aborting before checkpoint update")
    return result


def _prepare_metrics(path: Path, resume_step: int | None, resume_path: Path | None = None) -> None:
    """Refuse accidental run mixing or trim logs to an exact resume point."""
    run_manifest = path.parent / "run.json"
    if resume_step is None:
        if path.exists() or run_manifest.exists():
            raise FileExistsError(
                f"run output already exists in {path.parent}; choose a new out_dir or resume"
            )
        return
    if not path.exists():
        return
    last_checkpoint = path.parent / "last.pt"
    if (
        last_checkpoint.exists()
        and resume_path is not None
        and last_checkpoint.resolve() != resume_path.resolve()
    ):
        raise ValueError("refusing to rewind while a different last.pt exists in out_dir")
    retained = []
    future_records = False
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record["step"]) <= resume_step:
            retained.append(line)
        else:
            future_records = True
    best_checkpoint = path.parent / "best.pt"
    if future_records and best_checkpoint.exists():
        archive = path.parent / f"best.before-resume-{resume_step}.pt"
        if archive.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint archive {archive}")
        os.replace(best_checkpoint, archive)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""))
    os.replace(temporary, path)


def _load_data_manifest(data_dir: Path, config: TrainConfig) -> dict[str, Any]:
    path = data_dir / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing required data manifest: {path}")
    manifest = json.loads(path.read_text())
    if int(manifest["vocab_size"]) != config.model.vocab_size:
        raise ValueError("data tokenizer vocabulary does not match model vocab_size")
    for filename in ("train.bin", "val.bin"):
        expected = manifest.get("files", {}).get(filename)
        if not isinstance(expected, dict):
            raise TypeError(f"data manifest is missing identity for {filename}")
        token_path = data_dir / filename
        if token_path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"data file size mismatch: {token_path}")
        if file_sha256(token_path) != expected["sha256"]:
            raise ValueError(f"data file hash mismatch: {token_path}")
    result = {"contents": manifest, "sha256": file_sha256(path)}
    return result


def train(config: TrainConfig, resume: str | Path | None = None) -> Path:
    device = resolve_device(config.device)
    seed_everything(config.seed)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    data_dir = Path(config.data_dir)
    data_manifest = _load_data_manifest(data_dir, config)
    train_batcher = TokenBatcher(
        data_dir / "train.bin",
        config.model.context_length,
        config.batch_size,
        config.seed,
        device,
    )
    val_batcher = TokenBatcher(
        data_dir / "val.bin",
        config.model.context_length,
        config.batch_size,
        config.seed + 1,
        device,
    )
    model = GPT(config.model).to(device)
    optimizer = model.configure_optimizer(
        config.learning_rate,
        config.weight_decay,
        (config.beta1, config.beta2),
        device.type,
    )
    first_step = 0
    best_val_loss = float("inf")
    initial_git_state = _git_state()
    if resume is not None:
        first_step, best_val_loss, initial_git_state = load_checkpoint(
            resume,
            model,
            optimizer,
            train_batcher,
            val_batcher,
            config,
            data_manifest,
        )
    raw_model = model
    if config.compile:
        model = torch.compile(model)
    if config.dtype == "bfloat16" and device.type == "cuda":
        autocast_context = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext
    output = Path(config.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    _prepare_metrics(
        metrics_path,
        first_step if resume is not None else None,
        Path(resume) if resume is not None else None,
    )
    run_manifest = {
        "config": config.to_dict(),
        "parameters": raw_model.parameter_count(),
        "git_state": _git_state(),
        "initial_git_state": initial_git_state,
        "resumed_from": str(Path(resume).resolve()) if resume is not None else None,
        "data_manifest": data_manifest,
        "torch": torch.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (output / "run.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    model.train()
    for step in range(first_step, config.max_steps):
        started = time.perf_counter()
        lr = learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            x, y = train_batcher.next()
            with autocast_context():
                _, loss = model(x, y)
                assert loss is not None
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at step {step + 1}")
                scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            accumulated_loss += loss.detach().float().item() / config.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), config.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        completed_step = step + 1
        metrics = {
            "event": "train",
            "step": completed_step,
            "train_loss": accumulated_loss,
            "learning_rate": lr,
            "grad_norm": float(grad_norm),
            "seconds": elapsed,
            "tokens_per_second": config.tokens_per_step / elapsed,
            "tokens_seen": completed_step * config.tokens_per_step,
        }
        if completed_step % config.log_interval == 0:
            print(json.dumps(metrics), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
        if completed_step % config.eval_interval == 0 or completed_step == config.max_steps:
            val_loss = evaluate(model, val_batcher, config.eval_batches, autocast_context)
            eval_metrics = {"event": "eval", "step": completed_step, "val_loss": val_loss}
            print(json.dumps(eval_metrics), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_metrics) + "\n")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    output / "best.pt",
                    raw_model,
                    optimizer,
                    config,
                    completed_step,
                    best_val_loss,
                    train_batcher,
                    val_batcher,
                    data_manifest,
                    initial_git_state,
                )
        if completed_step % config.checkpoint_interval == 0 or completed_step == config.max_steps:
            save_checkpoint(
                output / "last.pt",
                raw_model,
                optimizer,
                config,
                completed_step,
                best_val_loss,
                train_batcher,
                val_batcher,
                data_manifest,
                initial_git_state,
            )
    return output / "last.pt"
