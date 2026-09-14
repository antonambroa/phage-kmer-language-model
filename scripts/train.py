#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import math
import os
from functools import partial
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from phage_kmer_lm.checkpoint import save_checkpoint
from phage_kmer_lm.data import TokenWindowDataset, collate_next_token
from phage_kmer_lm.model import CausalKmerTransformer, ModelConfig
from phage_kmer_lm.tokenizer import KmerTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Train a causal k-mer Transformer baseline.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8, help="Per-process batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank = 0
        local_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return distributed, rank, local_rank, device


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    distributed, rank, local_rank, device = distributed_context()

    data_dir = Path(args.data_dir)
    tokenizer = KmerTokenizer.load(data_dir / "vocab.json")
    train_ds = TokenWindowDataset(data_dir / "train.pt")
    sample_payload = torch.load(data_dir / "train.pt", map_location="cpu", weights_only=False)
    context_length = int(sample_payload["metadata"]["context_length"])

    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if distributed else None
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=partial(collate_next_token, pad_id=tokenizer.pad_id),
    )

    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=context_length,
    )
    model = CausalKmerTransformer(config).to(device)
    wrapped = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None) if distributed else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        total_loss = 0.0
        steps = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
        if rank == 0:
            avg = total_loss / max(steps, 1)
            print(f"epoch={epoch + 1} loss={avg:.4f} perplexity={math.exp(min(avg, 20)):.2f}")

    if rank == 0:
        underlying = wrapped.module if isinstance(wrapped, DDP) else wrapped
        save_checkpoint(args.output, underlying, tokenizer, training_args=vars(args))
        print(f"checkpoint={args.output}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
