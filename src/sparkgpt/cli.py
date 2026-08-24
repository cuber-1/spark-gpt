"""Command-line interface for data preparation, training, and diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from sparkgpt import __version__
from sparkgpt.config import load_config
from sparkgpt.data import (
    load_tokenizer,
    prepare_token_files,
    train_sentencepiece,
)
from sparkgpt.model import GPT
from sparkgpt.training import resolve_device, train


def _human_count(value: int) -> str:
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000:
            return f"{value:.2f}{suffix}" if suffix else str(value)
        value /= 1000
    return f"{value:.2f}P"


def command_doctor(_: argparse.Namespace) -> int:
    cuda = torch.cuda.is_available()
    disk = shutil.disk_usage(Path.cwd())
    report = {
        "spark_gpt": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda else None,
        "bfloat16_supported": torch.cuda.is_bf16_supported() if cuda else False,
        "unified_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if cuda else None
        ),
        "disk_free_bytes": disk.free,
    }
    print(json.dumps(report, indent=2))
    if not cuda:
        print("warning: CUDA is unavailable; only smoke tests are practical", file=sys.stderr)
    elif not report["bfloat16_supported"]:
        print("warning: this CUDA device does not report BF16 support", file=sys.stderr)
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    parameters = config.model.parameter_count()
    print(
        json.dumps(
            {
                "parameters": parameters,
                "parameters_human": _human_count(parameters),
                "tokens_per_step": config.tokens_per_step,
                "planned_tokens": config.tokens_per_step * config.max_steps,
                "model": config.model.__dict__,
            },
            indent=2,
        )
    )
    return 0


def command_train_tokenizer(args: argparse.Namespace) -> int:
    model_path = train_sentencepiece(args.input, args.output_prefix, args.vocab_size)
    print(model_path)
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer)
    manifest = prepare_token_files(
        args.input,
        args.output,
        tokenizer,
        val_fraction=args.val_fraction,
        seed=args.seed,
        jsonl_field=args.jsonl_field,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    checkpoint = train(load_config(args.config), resume=args.resume)
    print(f"checkpoint: {checkpoint}")
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = resolve_device(config.device)
    if device.type != "cuda":
        raise RuntimeError("the training benchmark requires CUDA")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.set_float32_matmul_precision("high")
    sequence = args.sequence_length or config.model.context_length
    if sequence > config.model.context_length:
        raise ValueError("benchmark sequence exceeds the configured context length")
    model = GPT(config.model).to(device)
    parameter_count = model.parameter_count()
    optimizer = model.configure_optimizer(
        config.learning_rate,
        config.weight_decay,
        (config.beta1, config.beta2),
        device.type,
    )
    if args.compile:
        model = torch.compile(model)
    x = torch.randint(config.model.vocab_size, (args.batch_size, sequence), device=device)
    autocast_context = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if config.dtype == "bfloat16"
        else nullcontext
    )
    torch.cuda.reset_peak_memory_stats(device)
    measurements = []
    total_steps = args.warmup_steps + args.steps
    for index in range(total_steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            _, loss = model(x, x)
        assert loss is not None
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= args.warmup_steps:
            measurements.append(elapsed)
    mean_seconds = sum(measurements) / len(measurements)
    tokens = args.batch_size * sequence
    print(
        json.dumps(
            {
                "parameters": parameter_count,
                "batch_size": args.batch_size,
                "sequence_length": sequence,
                "steps": args.steps,
                "seconds_per_step": mean_seconds,
                "tokens_per_second": tokens / mean_seconds,
                "estimated_training_tflops": 6 * parameter_count * tokens / mean_seconds / 1e12,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(device),
            },
            indent=2,
        )
    )
    return 0


def command_generate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    device = resolve_device(config.device)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.vocab_size != config.model.vocab_size:
        raise ValueError("tokenizer vocabulary does not match the model configuration")
    model = GPT(config.model)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"]["model"] != config.model.__dict__:
        raise ValueError("checkpoint model configuration does not match")
    model.load_state_dict(checkpoint["model"])
    expected_tokenizer = checkpoint.get("data_manifest", {}).get("contents", {}).get("tokenizer")
    if expected_tokenizer is None:
        raise ValueError("checkpoint does not contain tokenizer identity metadata")
    actual_tokenizer = tokenizer.metadata()
    for key in ("kind", "identity", "sha256", "vocab_size"):
        if key in expected_tokenizer and expected_tokenizer[key] != actual_tokenizer.get(key):
            raise ValueError(f"tokenizer identity mismatch for {key}")
    del checkpoint
    model.to(device)
    model.eval()
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda" and config.dtype == "bfloat16",
    ):
        output = model.generate(
            prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    print(tokenizer.decode(output[0].tolist()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spark-gpt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report training environment readiness")
    doctor.set_defaults(handler=command_doctor)

    inspect = subparsers.add_parser("inspect", help="show config size without allocating a model")
    inspect.add_argument("--config", required=True)
    inspect.set_defaults(handler=command_inspect)

    tokenizer = subparsers.add_parser("train-tokenizer", help="train a SentencePiece BPE")
    tokenizer.add_argument("--input", nargs="+", required=True)
    tokenizer.add_argument("--output-prefix", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=32768)
    tokenizer.set_defaults(handler=command_train_tokenizer)

    prepare = subparsers.add_parser("prepare", help="tokenize text/JSONL into packed files")
    prepare.add_argument("--input", nargs="+", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--tokenizer", default="byte")
    prepare.add_argument("--jsonl-field", default="text")
    prepare.add_argument("--val-fraction", type=float, default=0.002)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(handler=command_prepare)

    train_parser = subparsers.add_parser("train", help="train or resume a model")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--resume")
    train_parser.set_defaults(handler=command_train)

    benchmark = subparsers.add_parser("benchmark", help="measure synthetic train throughput")
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--sequence-length", type=int)
    benchmark.add_argument("--warmup-steps", type=int, default=2)
    benchmark.add_argument("--steps", type=int, default=5)
    benchmark.add_argument("--compile", action="store_true")
    benchmark.set_defaults(handler=command_benchmark)

    generate = subparsers.add_parser("generate", help="sample from a checkpoint")
    generate.add_argument("--config", required=True)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--tokenizer", default="byte")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.set_defaults(handler=command_generate)
    return parser


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
