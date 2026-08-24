#!/usr/bin/env python3
"""Stream and tokenize a pinned FineWeb-Edu revision into SparkGPT files."""

from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

from sparkgpt.data import SentencePieceTokenizer, file_sha256


def atomic_json(path: Path, contents: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(contents, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def local_source_files(root: Path) -> list[Path]:
    files = sorted(path.resolve() for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no Parquet files found under {root}")
    return files


def source_manifest(root: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root.resolve())),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in files
    ]


def finish_local_parquet(
    *,
    state_path: Path,
    meta_path: Path,
    partial_train: Path,
    partial_val: Path,
    final_train: Path,
    final_val: Path,
    identity: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finish or recover the two-file commit after hashes have been recorded durably."""
    if manifest is not None:
        atomic_json(
            state_path,
            {
                "phase": "finalizing",
                "identity": identity,
                "manifest": manifest,
            },
        )
    else:
        state = json.loads(state_path.read_text())
        if state.get("phase") != "finalizing" or state.get("identity") != identity:
            raise ValueError("invalid finalization state")
        manifest = state["manifest"]

    assert manifest is not None
    for name, partial, final in (
        ("train.bin", partial_train, final_train),
        ("val.bin", partial_val, final_val),
    ):
        expected = manifest["files"][name]
        if partial.exists() and final.exists():
            raise FileExistsError(f"both partial and final files exist for {name}")
        candidate = final if final.exists() else partial
        if not candidate.exists():
            raise FileNotFoundError(f"missing both partial and final files for {name}")
        if candidate.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name} byte count changed during finalization")
        if file_sha256(candidate) != expected["sha256"]:
            raise ValueError(f"{name} hash changed during finalization")
        if candidate == partial:
            os.replace(partial, final)

    if meta_path.exists():
        if json.loads(meta_path.read_text()) != manifest:
            raise ValueError("existing metadata does not match finalization state")
    else:
        atomic_json(meta_path, manifest)
    state_path.unlink()
    directory_fd = os.open(meta_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return manifest


def prepare_local_parquet(
    *,
    root: Path,
    output: Path,
    tokenizer: SentencePieceTokenizer,
    dataset: str,
    dataset_config: str,
    revision: str,
    max_tokens: int,
    val_fraction: float,
    seed: int,
    progress_documents: int,
    parquet_batch_size: int,
) -> dict[str, Any]:
    """Pack local Parquet shards, resuming only from fsynced shard boundaries."""
    import pyarrow.parquet as pq

    lock_path = output / "prepare.lock"
    _lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(_lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"another packer owns {lock_path}") from error

    files = local_source_files(root)
    sources = source_manifest(root, files)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    final_train = output / "train.bin"
    final_val = output / "val.bin"
    partial_train = output / "train.bin.partial"
    partial_val = output / "val.bin.partial"
    state_path = output / "resume.json"
    meta_path = output / "meta.json"

    identity = {
        "format": "fineweb-local-pack-v1",
        "dataset": dataset,
        "dataset_config": dataset_config,
        "revision": revision,
        "source_root": str(root.resolve()),
        "sources": sources,
        "max_tokens": max_tokens,
        "val_fraction": val_fraction,
        "seed": seed,
        "dtype": np.dtype(dtype).name,
        "tokenizer": tokenizer.metadata(),
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("resume state does not match source, tokenizer, or packing arguments")
        if state.get("phase") == "finalizing":
            return finish_local_parquet(
                state_path=state_path,
                meta_path=meta_path,
                partial_train=partial_train,
                partial_val=partial_val,
                final_train=final_train,
                final_val=final_val,
                identity=identity,
            )
        if state.get("phase", "packing") != "packing":
            raise ValueError("unknown resume phase")
        if any(path.exists() for path in (final_train, final_val, meta_path)):
            raise FileExistsError("final output exists while packing state is incomplete")
        if not partial_train.exists() or not partial_val.exists():
            raise FileNotFoundError("resume state exists without both partial token files")
        counts = dict(state["counts"])
        itemsize = np.dtype(dtype).itemsize
        for split, partial in (("train", partial_train), ("val", partial_val)):
            saved_bytes = int(state[f"{split}_bytes"])
            if saved_bytes != int(counts[split]) * itemsize:
                raise ValueError(f"{split} resume byte count disagrees with token count")
            if partial.stat().st_size < saved_bytes:
                raise ValueError(f"{split} partial file is shorter than its resume boundary")
            with partial.open("r+b") as handle:
                handle.truncate(saved_bytes)
        next_shard = int(state["next_shard"])
        print(
            json.dumps(
                {
                    "event": "resume",
                    "next_shard": next_shard,
                    **counts,
                }
            ),
            flush=True,
        )
    else:
        if any(path.exists() for path in (final_train, final_val, meta_path)):
            raise FileExistsError(f"refusing to overwrite completed data in {output}")
        if partial_train.exists() or partial_val.exists():
            raise FileExistsError("partial token files exist without a verified resume state")
        partial_train.touch()
        partial_val.touch()
        counts = {"train": 0, "val": 0, "documents": 0}
        next_shard = 0
        atomic_json(
            state_path,
            {
                "phase": "packing",
                "identity": identity,
                "next_shard": next_shard,
                "counts": counts,
                "train_bytes": 0,
                "val_bytes": 0,
            },
        )

    started = time.monotonic()
    complete = False
    with partial_train.open("ab") as train_handle, partial_val.open("ab") as val_handle:
        for shard_index in range(next_shard, len(files)):
            parquet_file = pq.ParquetFile(files[shard_index])
            for batch in parquet_file.iter_batches(
                batch_size=parquet_batch_size,
                columns=["text"],
            ):
                texts = [
                    text for text in batch.column(0).to_pylist() if isinstance(text, str) and text
                ]
                for text, token_ids in zip(texts, tokenizer.encode_batch(texts), strict=True):
                    if tokenizer.eos_id is not None:
                        token_ids.append(tokenizer.eos_id)
                    total = counts["train"] + counts["val"]
                    if total + len(token_ids) > max_tokens:
                        complete = True
                        break
                    split = "val" if validation_document(text, val_fraction, seed) else "train"
                    np.asarray(token_ids, dtype=dtype).tofile(
                        val_handle if split == "val" else train_handle
                    )
                    counts[split] += len(token_ids)
                    counts["documents"] += 1
                    if counts["documents"] % progress_documents == 0:
                        total = counts["train"] + counts["val"]
                        elapsed = time.monotonic() - started
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "shard": shard_index,
                                    **counts,
                                    "tokens": total,
                                    "tokens_per_second": total / elapsed,
                                }
                            ),
                            flush=True,
                        )
                if complete:
                    break
            train_handle.flush()
            val_handle.flush()
            os.fsync(train_handle.fileno())
            os.fsync(val_handle.fileno())
            if complete:
                break
            next_shard = shard_index + 1
            atomic_json(
                state_path,
                {
                    "phase": "packing",
                    "identity": identity,
                    "next_shard": next_shard,
                    "counts": counts,
                    "train_bytes": train_handle.tell(),
                    "val_bytes": val_handle.tell(),
                },
            )

    if not complete and counts["train"] + counts["val"] < max_tokens:
        raise RuntimeError(
            f"local snapshot ended at {counts['train'] + counts['val']} tokens, "
            f"below requested {max_tokens}"
        )
    manifest = {
        "format": "raw-token-ids-v1",
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "revision": revision,
        "split": "train",
        "license": "ODC-By-1.0",
        "upstream_terms": "https://commoncrawl.org/terms-of-use",
        "source_root": str(root.resolve()),
        "sources": sources,
        "seed": seed,
        "val_fraction": val_fraction,
        "max_tokens": max_tokens,
        "tokenizer": tokenizer.metadata(),
        **counts,
        "files": {
            "train.bin": {
                "bytes": partial_train.stat().st_size,
                "sha256": file_sha256(partial_train),
            },
            "val.bin": {
                "bytes": partial_val.stat().st_size,
                "sha256": file_sha256(partial_val),
            },
        },
    }
    return finish_local_parquet(
        state_path=state_path,
        meta_path=meta_path,
        partial_train=partial_train,
        partial_val=partial_val,
        final_train=final_train,
        final_val=final_val,
        identity=identity,
        manifest=manifest,
    )


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
    parser.add_argument(
        "--local-parquet-root",
        help="pack a resumable, already-downloaded Parquet snapshot instead of streaming",
    )
    parser.add_argument("--max-tokens", type=int, default=4_600_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--progress-documents", type=int, default=1_000)
    parser.add_argument("--parquet-batch-size", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val-fraction must be between zero and one")
    if args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    if args.progress_documents <= 0 or args.parquet_batch_size <= 0:
        raise ValueError("progress-documents and parquet-batch-size must be positive")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if args.local_parquet_root:
        if not args.revision:
            raise ValueError("--revision is required with --local-parquet-root")
        manifest = prepare_local_parquet(
            root=Path(args.local_parquet_root),
            output=output,
            tokenizer=tokenizer,
            dataset=args.dataset,
            dataset_config=args.dataset_config,
            revision=args.revision,
            max_tokens=args.max_tokens,
            val_fraction=args.val_fraction,
            seed=args.seed,
            progress_documents=args.progress_documents,
            parquet_batch_size=args.parquet_batch_size,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    final_train = output / "train.bin"
    final_val = output / "val.bin"
    partial_train = output / "train.bin.partial"
    partial_val = output / "val.bin.partial"
    meta_path = output / "meta.json"
    if any(
        path.exists() for path in (final_train, final_val, partial_train, partial_val, meta_path)
    ):
        raise FileExistsError(f"refusing to overwrite existing or partial data in {output}")
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
