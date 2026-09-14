from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DNA_ALPHABET = "ACGT"
SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")


@dataclass(frozen=True)
class KmerTokenizer:
    """Tokenizer for stride-1 DNA k-mers.

    The vocabulary is deterministic: special tokens first, followed by all A/C/G/T
    k-mers in lexicographic product order. Ambiguous k-mers map to ``<UNK>``.
    """

    k: int
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, k: int = 6) -> "KmerTokenizer":
        if k < 1:
            raise ValueError("k must be >= 1")
        tokens = list(SPECIAL_TOKENS)
        tokens.extend("".join(p) for p in itertools.product(DNA_ALPHABET, repeat=k))
        return cls(k=k, token_to_id={token: idx for idx, token in enumerate(tokens)})

    @property
    def id_to_token(self) -> dict[int, str]:
        return {idx: token for token, idx in self.token_to_id.items()}

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> list[int]:
        sequence = sequence.upper().replace(" ", "").replace("\n", "")
        ids: list[int] = []
        if add_special_tokens:
            ids.append(self.bos_id)
        if len(sequence) >= self.k:
            for i in range(len(sequence) - self.k + 1):
                kmer = sequence[i : i + self.k]
                ids.append(self.token_to_id.get(kmer, self.unk_id))
        if add_special_tokens:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Iterable[int], strict_overlap: bool = False) -> str:
        id_to_token = self.id_to_token
        kmers = [
            id_to_token[int(idx)]
            for idx in token_ids
            if int(idx) in id_to_token and id_to_token[int(idx)] not in SPECIAL_TOKENS
        ]
        if not kmers:
            return ""
        sequence = kmers[0]
        previous = kmers[0]
        for kmer in kmers[1:]:
            overlap_ok = previous[1:] == kmer[:-1]
            if strict_overlap and not overlap_ok:
                raise ValueError(f"Non-overlapping transition: {previous} -> {kmer}")
            sequence += kmer[-1]
            previous = kmer
        return sequence

    def compatible_next_ids(self, previous_id: int) -> list[int]:
        """Return regular k-mer IDs compatible with a stride-1 transition."""
        token = self.id_to_token.get(int(previous_id))
        if token is None or token in SPECIAL_TOKENS:
            return [
                idx
                for text, idx in self.token_to_id.items()
                if text not in SPECIAL_TOKENS
            ]
        suffix = token[1:]
        return [self.token_to_id[suffix + base] for base in DNA_ALPHABET]

    def save(self, path: str | Path) -> None:
        payload = {"k": self.k, "token_to_id": self.token_to_id}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "KmerTokenizer":
        payload = json.loads(Path(path).read_text())
        return cls(k=int(payload["k"]), token_to_id={k: int(v) for k, v in payload["token_to_id"].items()})
