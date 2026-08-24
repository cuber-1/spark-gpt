#!/usr/bin/env python3
"""Stream and tokenize a pinned FineWeb-Edu revision into SparkGPT files."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

from sparkgpt.data import SentencePieceTokenizer, file_sha256


def shutdown_fsspec_loop() -> None:
    """Join fsspec's global async thread before CPython extension teardown."""
    import fsspec.asyn

    loop = fsspec.asyn.loop[0]
    thread = fsspec.asyn.iothread[0]
    if loop is not None and thread is not None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)


def validation_document(text: str, val_fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=8, person=seed.to_bytes(8, "little")
    ).digest()
    return int.from_bytes(digest, "little") / 2**64 < val_fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", default="data/fineweb_edu")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--revision")
    parser.add_argument("--max-tokens", type=int, default=4_600_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--progress-documents", type=int, default=1_000)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    final_train = output / "train.bin"
    final_val = output / "val.bin"
    partial_train = output / "train.bin.partial"
    partial_val = output / "val.bin.partial"
    meta_path = output / "meta.json"
    if any(
        path.exists() for path in (final_train, final_val, partial_train, partial_val, meta_path)
    ):
        raise FileExistsError(f"refusing to overwrite existing or partial data in {output}")
    tokenizer_path = Path(args.tokenizer)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    revision = args.revision or HfApi().dataset_info(args.dataset).sha
    dataset = load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        revision=revision,
        streaming=True,
    )
    counts = {"train": 0, "val": 0, "documents": 0}
    started = time.monotonic()
    iterator = iter(dataset)
    try:
        with partial_train.open("wb") as train_handle, partial_val.open("wb") as val_handle:
            for row in iterator:
                text = row.get("text")
                if not isinstance(text, str) or not text:
                    continue
                token_ids = tokenizer.encode(text)
                if tokenizer.eos_id is not None:
                    token_ids.append(tokenizer.eos_id)
                if not token_ids:
                    continue
                if counts["train"] + counts["val"] + len(token_ids) > args.max_tokens:
                    break
                split = (
                    "val" if validation_document(text, args.val_fraction, args.seed) else "train"
                )
                np.asarray(token_ids, dtype=dtype).tofile(
                    val_handle if split == "val" else train_handle
                )
                counts[split] += len(token_ids)
                counts["documents"] += 1
                if counts["documents"] % args.progress_documents == 0:
                    total = counts["train"] + counts["val"]
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            {
                                **counts,
                                "tokens": total,
                                "tokens_per_second": total / elapsed,
                            }
                        ),
                        flush=True,
                    )
            train_handle.flush()
            val_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(val_handle.fileno())
        partial_train.replace(final_train)
        partial_val.replace(final_val)
    except BaseException:
        print(f"partial files retained in {output} for diagnosis", flush=True)
        raise
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    manifest = {
        "format": "raw-token-ids-v1",
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "revision": revision,
        "split": "train",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "max_tokens": args.max_tokens,
        "tokenizer": tokenizer.metadata(),
        **counts,
        "files": {
            "train.bin": {
                "bytes": final_train.stat().st_size,
                "sha256": file_sha256(final_train),
            },
            "val.bin": {
                "bytes": final_val.stat().st_size,
                "sha256": file_sha256(final_val),
            },
        },
    }
    partial_meta = output / "meta.json.partial"
    partial_meta.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial_meta, meta_path)
    print(json.dumps(manifest, indent=2))
    del iterator, dataset
    gc.collect()
    shutdown_fsspec_loop()
    return 0


if __name__ == "__main__":
    exit_code = main()
    # See huggingface/datasets#7566. Use only when the native streaming thread
    # crashes after successful output; explicit flushes precede the clean exit.
    if exit_code == 0 and os.environ.get("SPARKGPT_EXIT_WITHOUT_FINALIZE") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    raise SystemExit(exit_code)
