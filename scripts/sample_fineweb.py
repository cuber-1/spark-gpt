#!/usr/bin/env python3
"""Stream a bounded FineWeb-Edu sample for SentencePiece training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


def shutdown_fsspec_loop() -> None:
    """Join fsspec's global async thread before CPython extension teardown."""
    import fsspec.asyn

    loop = fsspec.asyn.loop[0]
    thread = fsspec.asyn.iothread[0]
    if loop is not None and thread is not None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/tokenizer_sample.txt")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--revision")
    parser.add_argument("--max-characters", type=int, default=50_000_000)
    parser.add_argument("--line-characters", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    dataset = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        revision=revision,
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=10_000)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or temporary.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sample files for {output}")
    characters = 0
    documents = 0
    digest = hashlib.sha256()
    iterator = iter(dataset)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in iterator:
                text = " ".join(str(row["text"]).split())
                if not text:
                    continue
                remaining = args.max_characters - characters
                if remaining <= 0:
                    break
                text = text[:remaining]
                for start in range(0, len(text), args.line_characters):
                    line = text[start : start + args.line_characters] + "\n"
                    handle.write(line)
                    digest.update(line.encode("utf-8"))
                characters += len(text)
                documents += 1
                if characters >= args.max_characters:
                    break
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    temporary.replace(output)
    manifest = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "revision": revision,
        "split": "train",
        "seed": args.seed,
        "documents": documents,
        "characters": characters,
        "line_characters": args.line_characters,
        "sha256": digest.hexdigest(),
        "purpose": "SentencePiece tokenizer training sample",
    }
    partial_manifest = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    partial_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    partial_manifest.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    # Close streaming/download iterators before Python begins interpreter teardown.
    del iterator, dataset
    gc.collect()
    shutdown_fsspec_loop()
    return 0


if __name__ == "__main__":
    exit_code = main()
    # Hugging Face streaming can leave an unreachable native thread that crashes
    # during interpreter teardown (huggingface/datasets#7566). This opt-in mirrors
    # NVIDIA Cosmos: all artifacts are already closed/renamed before bypassing atexit.
    if exit_code == 0 and os.environ.get("SPARKGPT_EXIT_WITHOUT_FINALIZE") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    raise SystemExit(exit_code)
