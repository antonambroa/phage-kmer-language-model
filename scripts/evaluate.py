#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import math
from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phage_kmer_lm.checkpoint import load_checkpoint
from phage_kmer_lm.data import TokenWindowDataset, collate_next_token


def main():
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint on token windows.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    dataset = TokenWindowDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=partial(collate_next_token, pad_id=tokenizer.pad_id))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id, reduction="sum")

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += float(criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)))
            total_tokens += int(targets.ne(tokenizer.pad_id).sum())

    nll = total_loss / max(total_tokens, 1)
    print(f"tokens={total_tokens} nll={nll:.4f} perplexity={math.exp(min(nll, 20)):.2f}")


if __name__ == "__main__":
    main()
