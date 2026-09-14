from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    name = None
    chunks: list[str] = []
    with Path(path).open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            yield name, "".join(chunks)


def make_windows(token_ids: list[int], context_length: int, stride: int) -> list[list[int]]:
    """Split a tokenized record into fixed-size LM examples.

    Stored examples have ``context_length + 1`` tokens so that inputs and next-token
    targets can be formed by shifting by one position.
    """
    width = context_length + 1
    if width < 2 or stride < 1:
        raise ValueError("context_length and stride must be positive")
    if len(token_ids) <= width:
        return [token_ids]
    starts = list(range(0, len(token_ids) - width + 1, stride))
    if starts[-1] != len(token_ids) - width:
        starts.append(len(token_ids) - width)
    return [token_ids[start : start + width] for start in starts]


def split_examples(examples: list[list[int]], val_fraction: float, seed: int) -> tuple[list[list[int]], list[list[int]]]:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    items = list(examples)
    random.Random(seed).shuffle(items)
    n_val = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 0
    return items[n_val:], items[:n_val]


class TokenWindowDataset(Dataset):
    def __init__(self, path: str | Path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.examples = payload["examples"] if isinstance(payload, dict) else payload

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.examples[index], dtype=torch.long)


def collate_next_token(batch: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(item.numel() for item in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for row, item in enumerate(batch):
        padded[row, : item.numel()] = item
    return padded[:, :-1], padded[:, 1:]
