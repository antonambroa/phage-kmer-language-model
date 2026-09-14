#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import argparse

import torch

from phage_kmer_lm.checkpoint import load_checkpoint
from phage_kmer_lm.tokenizer import SPECIAL_TOKENS


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and 0 < top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1)].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate(model, tokenizer, max_new_bases: int, device, temperature: float, top_k: int | None, seed_sequence: str | None):
    if seed_sequence:
        ids = tokenizer.encode(seed_sequence, add_special_tokens=False)
        if not ids or ids[0] == tokenizer.unk_id:
            raise ValueError("seed sequence must contain at least one unambiguous k-mer")
        generated = ids
    else:
        generated = [tokenizer.bos_id]

    max_tokens = min(model.config.max_len, max_new_bases + tokenizer.k + 2)
    while len(generated) < max_tokens:
        context = torch.tensor([generated[-model.config.max_len :]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(context)[0, -1].clone()

        previous = generated[-1]
        previous_text = tokenizer.id_to_token.get(previous, "<UNK>")
        if previous_text not in SPECIAL_TOKENS:
            allowed = tokenizer.compatible_next_ids(previous) + [tokenizer.eos_id]
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = logits[allowed]
            logits = mask
        else:
            logits[tokenizer.pad_id] = float("-inf")
            logits[tokenizer.unk_id] = float("-inf")
            logits[tokenizer.bos_id] = float("-inf")

        next_id = sample_from_logits(logits, temperature, top_k)
        if next_id == tokenizer.eos_id:
            break
        generated.append(next_id)

    return tokenizer.decode(generated, strict_overlap=True)


def main():
    p = argparse.ArgumentParser(description="Generate a sequence from a trained k-mer language model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-sequences", type=int, default=5)
    p.add_argument("--max-new-bases", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--seed-sequence", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    with open(args.output, "w") as handle:
        for i in range(args.num_sequences):
            seq = generate(model, tokenizer, args.max_new_bases, device, args.temperature, args.top_k, args.seed_sequence)
            handle.write(f">generated_{i + 1}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")
    print(f"wrote={args.num_sequences} output={args.output}")


if __name__ == "__main__":
    main()
