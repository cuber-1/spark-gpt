"""Tokenizer helpers and deterministic packed-token loading."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class Tokenizer(Protocol):
    vocab_size: int
    eos_id: int | None

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def metadata(self) -> dict[str, object]: ...


class ByteTokenizer:
    """A dependency-free tokenizer for tests and small learning runs."""

    vocab_size = 256
    eos_id = None

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    @staticmethod
    def metadata() -> dict[str, object]:
        return {"kind": "byte", "identity": "utf8-bytes-v1", "vocab_size": 256}


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise RuntimeError(
                "SentencePiece support requires: pip install 'spark-gpt[data]'"
            ) from error
        self.model_path = Path(model_path).resolve()
        self._processor = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.vocab_size = self._processor.vocab_size()
        eos_id = self._processor.eos_id()
        self.eos_id = eos_id if eos_id >= 0 else None

    def encode(self, text: str) -> list[int]:
        return list(self._processor.encode(text, out_type=int))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._processor.decode(list(token_ids))

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "sentencepiece",
            "path": str(self.model_path),
            "sha256": file_sha256(self.model_path),
            "vocab_size": self.vocab_size,
            "eos_id": self.eos_id,
        }


def load_tokenizer(name_or_path: str | Path) -> Tokenizer:
    return ByteTokenizer() if str(name_or_path) == "byte" else SentencePieceTokenizer(name_or_path)


def train_sentencepiece(
    input_paths: Sequence[str | Path], output_prefix: str | Path, vocab_size: int
) -> Path:
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError(
            "SentencePiece support requires: pip install 'spark-gpt[data]'"
        ) from error
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if prefix.with_suffix(".model").exists() or prefix.with_suffix(".vocab").exists():
        raise FileExistsError(f"refusing to overwrite tokenizer files for {prefix}")
    spm.SentencePieceTrainer.train(
        input=",".join(str(Path(path)) for path in input_paths),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        max_sentence_length=16_384,
        normalization_rule_name="nmt_nfkc_cf",
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=-1,
        shuffle_input_sentence=True,
    )
    return prefix.with_suffix(".model")


def iter_documents(paths: Sequence[str | Path], jsonl_field: str) -> Iterator[str]:
    """Yield one document per JSONL row or one document per non-empty text line."""
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix == ".jsonl":
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row.get(jsonl_field)
                    if not isinstance(text, str):
                        raise TypeError(f"{path}:{line_number}: {jsonl_field!r} is not a string")
                    if text:
                        yield text
            else:
                for line in handle:
                    text = line.strip()
                    if text:
                        yield text


def _validation_document(text: str, val_fraction: float, seed: int) -> bool:
    digest = hashlib.blake2b(
        text.encode("utf-8"), digest_size=8, person=seed.to_bytes(8, "little")
    ).digest()
    bucket = int.from_bytes(digest, "little") / 2**64
    return bucket < val_fraction


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_token_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    tokenizer: Tokenizer,
    val_fraction: float = 0.002,
    seed: int = 1337,
    jsonl_field: str = "text",
) -> dict[str, object]:
    """Tokenize documents into raw memory-mapped train/validation files."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= np.iinfo(np.uint16).max else np.uint32
    counts = {"train": 0, "val": 0, "documents": 0}
    train_path = output / "train.bin"
    val_path = output / "val.bin"
    meta_path = output / "meta.json"
    partial_train = output / "train.bin.partial"
    partial_val = output / "val.bin.partial"
    if any(path.exists() for path in (train_path, val_path, meta_path, partial_train, partial_val)):
        raise FileExistsError(f"refusing to overwrite existing or partial data in {output}")
    with partial_train.open("wb") as train_handle, partial_val.open("wb") as val_handle:
        for document in iter_documents(input_paths, jsonl_field=jsonl_field):
            token_ids = tokenizer.encode(document)
            if tokenizer.eos_id is not None:
                token_ids.append(tokenizer.eos_id)
            if not token_ids:
                continue
            split = "val" if _validation_document(document, val_fraction, seed) else "train"
            handle = val_handle if split == "val" else train_handle
            np.asarray(token_ids, dtype=dtype).tofile(handle)
            counts[split] += len(token_ids)
            counts["documents"] += 1
        train_handle.flush()
        val_handle.flush()
        os.fsync(train_handle.fileno())
        os.fsync(val_handle.fileno())
    os.replace(partial_train, train_path)
    os.replace(partial_val, val_path)
    manifest: dict[str, object] = {
        "format": "raw-token-ids-v1",
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "seed": seed,
        "val_fraction": val_fraction,
        "jsonl_field": jsonl_field,
        "tokenizer": tokenizer.metadata(),
        "inputs": [
            {"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in input_paths
        ],
        **counts,
        "files": {
            "train.bin": {
                "bytes": train_path.stat().st_size,
                "sha256": file_sha256(train_path),
            },
            "val.bin": {
                "bytes": val_path.stat().st_size,
                "sha256": file_sha256(val_path),
            },
        },
    }
    partial_meta = output / "meta.json.partial"
    partial_meta.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial_meta, meta_path)
    return manifest


class TokenBatcher:
    """Deterministically sample next-token batches from a packed token file."""

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        batch_size: int,
        seed: int,
        device: torch.device,
    ) -> None:
        path = Path(path)
        manifest_path = path.parent / "meta.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        dtype = np.dtype(manifest.get("dtype", "uint16"))
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        self.context_length = context_length
        self.batch_size = batch_size
        self.device = device
        self.num_windows = (len(self.tokens) - 1) // context_length
        if self.num_windows < batch_size:
            raise ValueError(
                f"{path} has {len(self.tokens)} tokens; need at least "
                f"{batch_size * context_length + 1} for one full batch"
            )
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self._offsets = np.arange(context_length + 1, dtype=np.int64)
        self.permutation = torch.randperm(self.num_windows, generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cursor + self.batch_size > self.num_windows:
            self.permutation = torch.randperm(self.num_windows, generator=self.generator)
            self.cursor = 0
            self.epoch += 1
        window_ids = self.permutation[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        starts = (window_ids * self.context_length).numpy()
        indices = starts[:, None] + self._offsets[None, :]
        batch = torch.from_numpy(np.asarray(self.tokens[indices], dtype=np.int64))
        x, y = batch[:, :-1], batch[:, 1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation,
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["generator_state"])
        permutation = state["permutation"]
        if len(permutation) != self.num_windows:
            raise ValueError("token file window count changed since checkpoint")
        self.permutation = permutation
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
