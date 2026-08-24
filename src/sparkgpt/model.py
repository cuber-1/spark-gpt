"""A small, readable decoder-only Transformer."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from sparkgpt.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(dtype=x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = q.size(-2)
        positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)
        cos = angles.cos().to(dtype=q.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=q.dtype)[None, None, :, :]
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.kv_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.kv_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        q = self.q_proj(x).view(batch, sequence, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, sequence, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, sequence, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        if self.n_kv_head != self.n_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.out_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = SwiGLU(config)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + F.dropout(self.attention(self.attention_norm(x)), self.dropout, self.training)
        return x + F.dropout(self.mlp(self.mlp_norm(x)), self.dropout, self.training)


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd, config.norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(
                block.attention.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layer),
            )
            nn.init.normal_(
                block.mlp.down_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layer),
            )
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        if token_ids.size(1) > self.config.context_length:
            raise ValueError(
                f"sequence length {token_ids.size(1)} exceeds context length "
                f"{self.config.context_length}"
            )
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizer(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in parameters.values() if p.dim() >= 2]
        no_decay = [p for p in parameters.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
            fused=fused_available and device_type == "cuda",
        )

    @torch.inference_mode()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < values[:, [-1]]] = -float("inf")
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def config_dict(self) -> dict[str, object]:
        return asdict(self.config)
