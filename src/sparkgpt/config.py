"""Typed configuration and parameter accounting."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    intermediate_size: int
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    bias: bool = False
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "context_length",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "intermediate_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_head * self.head_dim

    def parameter_count(self) -> int:
        """Return the exact parameter count for :class:`sparkgpt.model.GPT`."""
        d = self.n_embd
        h = self.intermediate_size
        kv = self.kv_dim
        embedding = self.vocab_size * d
        attention_weights = d * d + 2 * d * kv + d * d
        mlp_weights = 3 * d * h
        norms = 2 * d
        per_layer = attention_weights + mlp_weights + norms
        final_norm = d
        output = 0 if self.tie_embeddings else self.vocab_size * d
        biases = 0
        if self.bias:
            attention_biases = d + 2 * kv + d
            mlp_biases = 2 * h + d
            biases = self.n_layer * (attention_biases + mlp_biases)
        return embedding + self.n_layer * per_layer + final_norm + output + biases


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig
    data_dir: str = "data/demo"
    out_dir: str = "runs/default"
    seed: int = 1337
    device: str = "auto"
    dtype: str = "bfloat16"
    compile: bool = False
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 50
    eval_batches: int = 10
    log_interval: int = 1
    checkpoint_interval: int = 100

    def __post_init__(self) -> None:
        positive = (
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
            "eval_interval",
            "eval_batches",
            "log_interval",
            "checkpoint_interval",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16")
        if not 0.0 <= self.min_lr <= self.learning_rate:
            raise ValueError("min_lr must be between zero and learning_rate")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.model.context_length

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    """Load a TOML file with ``[model]`` and ``[training]`` tables."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown = set(raw) - {"model", "training"}
    if unknown:
        raise ValueError(f"unknown top-level config tables: {sorted(unknown)}")
    if "model" not in raw:
        raise ValueError("config is missing [model]")
    model = ModelConfig(**raw["model"])
    return TrainConfig(model=model, **raw.get("training", {}))
