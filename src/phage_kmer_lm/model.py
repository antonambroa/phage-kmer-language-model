from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    pad_id: int
    embed_dim: int = 128
    num_heads: int = 8
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    max_len: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        return cls(**payload)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even for sinusoidal positional encoding")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)]


class CausalKmerTransformer(nn.Module):
    """Decoder-style autoregressive model implemented with TransformerEncoder blocks.

    The causal attention mask prevents each position from seeing future tokens. A
    padding mask additionally removes padded keys from attention.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_id)
        self.positional = PositionalEncoding(config.embed_dim, config.max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.output = nn.Linear(config.embed_dim, config.vocab_size)

    @staticmethod
    def causal_mask(length: int, device: torch.device | None = None) -> torch.Tensor:
        # True means "do not attend" for TransformerEncoder boolean masks.
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        x = self.positional(self.embedding(token_ids))
        attn_mask = self.causal_mask(token_ids.size(1), token_ids.device)
        padding_mask = token_ids.eq(self.config.pad_id)
        x = self.transformer(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        return self.output(self.norm(x))
