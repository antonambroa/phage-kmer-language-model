from __future__ import annotations

from pathlib import Path

import torch

from .model import CausalKmerTransformer, ModelConfig
from .tokenizer import KmerTokenizer


def save_checkpoint(path: str | Path, model: CausalKmerTransformer, tokenizer: KmerTokenizer, **extra) -> None:
    payload = {
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "tokenizer": {"k": tokenizer.k, "token_to_id": tokenizer.token_to_id},
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> tuple[CausalKmerTransformer, KmerTokenizer, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig.from_dict(payload["model_config"])
    tokenizer = KmerTokenizer(
        k=int(payload["tokenizer"]["k"]),
        token_to_id={k: int(v) for k, v in payload["tokenizer"]["token_to_id"].items()},
    )
    model = CausalKmerTransformer(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, tokenizer, payload
