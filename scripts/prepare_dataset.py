#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse
import random
import torch

from phage_kmer_lm.data import make_windows, read_fasta
from phage_kmer_lm.tokenizer import KmerTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Tokenize FASTA records and build fixed-context training windows.")
    p.add_argument("--fasta", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = KmerTokenizer.build(args.k)
    tokenizer.save(out / "vocab.json")

    records: list[tuple[str, list[int]]] = []
    for name, sequence in read_fasta(args.fasta):
        ids = tokenizer.encode(sequence)
        if len(ids) >= 3:
            records.append((name, ids))

    if len(records) < 2:
        raise SystemExit("At least two usable FASTA records are required for a train/validation split")

    random.Random(args.seed).shuffle(records)
    n_val = max(1, round(len(records) * args.val_fraction))
    n_val = min(n_val, len(records) - 1)
    val_records = records[:n_val]
    train_records = records[n_val:]

    train = [window for _, ids in train_records for window in make_windows(ids, args.context_length, args.stride)]
    val = [window for _, ids in val_records for window in make_windows(ids, args.context_length, args.stride)]
    metadata = {
        "k": args.k,
        "context_length": args.context_length,
        "stride": args.stride,
        "seed": args.seed,
        "records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "split_level": "fasta_record",
    }
    torch.save({"examples": train, "metadata": metadata}, out / "train.pt")
    torch.save({"examples": val, "metadata": metadata}, out / "val.pt")
    print(
        f"records={len(records)} train_records={len(train_records)} val_records={len(val_records)} "
        f"train_windows={len(train)} val_windows={len(val)} vocab_size={tokenizer.vocab_size}"
    )


if __name__ == "__main__":
    main()
