# phage-kmer-language-model

[![tests](https://github.com/antonambroa/phage-kmer-language-model/actions/workflows/tests.yml/badge.svg)](https://github.com/antonambroa/phage-kmer-language-model/actions/workflows/tests.yml)

A small autoregressive Transformer baseline for modeling DNA as overlapping k-mers. The code grew out of experiments on bacteriophage genome collections and is kept deliberately compact so the tokenization, context construction and next-token objective remain easy to inspect.

This repository is about **sequence-model engineering**, not biological validation. Generated sequences are model outputs only and should not be interpreted as viable or functional phages.

## What changed from the original experiment

The first exploratory version had several pieces that could drift independently: training, evaluation and generation instantiated different model sizes; the Transformer could attend to future tokens; and full genomes were passed directly to self-attention. This version makes those choices explicit:

- a causal attention mask is always applied;
- training uses fixed-length token windows instead of whole genomes;
- train/validation splitting happens at FASTA-record level before windowing;
- model configuration and tokenizer are stored in the checkpoint;
- evaluation and generation reconstruct the model from that checkpoint;
- padding is masked in both attention and loss;
- stride-1 k-mer generation can enforce valid overlap between consecutive k-mers;
- single-process CPU/GPU and `torchrun`-based DDP use the same training entry point.

## Representation

For `k=6`, a DNA sequence is represented as overlapping 6-mers with stride 1. The vocabulary contains all `4^6 = 4096` canonical DNA k-mers plus four special tokens (`PAD`, `BOS`, `EOS`, `UNK`).

A sequence such as:

```text
ACGTACGT
```

with `k=3` becomes:

```text
<BOS> ACG CGT GTA TAC ACG CGT <EOS>
```

During generation, once a regular k-mer has been produced, the next regular token can be restricted to the four k-mers compatible with the previous `(k-1)`-base suffix. This preserves the overlap implied by the representation.

## Quick smoke test

Install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Prepare the tiny example dataset:

```bash
python scripts/prepare_dataset.py \
  --fasta examples/toy_sequences.fasta \
  --output-dir data/processed \
  --k 3 \
  --context-length 16 \
  --stride 8 \
  --val-fraction 0.25
```

Train a deliberately small CPU model:

```bash
python scripts/train.py \
  --data-dir data/processed \
  --output checkpoints/toy.pt \
  --epochs 2 \
  --batch-size 4 \
  --embed-dim 32 \
  --num-heads 4 \
  --hidden-dim 64 \
  --num-layers 2
```

Evaluate it:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/toy.pt \
  --data data/processed/val.pt
```

Generation is available as a model sanity check:

```bash
python scripts/generate.py \
  --checkpoint checkpoints/toy.pt \
  --output outputs/generated.fasta \
  --num-sequences 3 \
  --max-new-bases 100
```

## HPC / multi-GPU

`scripts/train.py` detects a distributed `torchrun` environment automatically. `slurm/train_ddp.sbatch` is a generic two-GPU template; cluster-specific module names, accounts and partitions are intentionally not included.

```bash
sbatch slurm/train_ddp.sbatch
```

For a direct launch outside a scheduler:

```bash
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --data-dir data/processed \
  --output checkpoints/kmer_lm.pt
```

## Repository layout

```text
src/phage_kmer_lm/   tokenizer, dataset helpers, model and checkpoint I/O
scripts/              preprocessing, training, evaluation and generation
slurm/                generic DDP submission template
examples/             tiny synthetic input used for smoke tests
tests/                causal-mask, tokenizer and data-window tests
```

## Limitations

This is a baseline rather than a claim that overlapping k-mers are the best representation for long viral genomes. Standard self-attention still scales quadratically with context length, so the code intentionally models local windows. The train/validation split is performed by FASTA record before windowing. For comparative benchmarking on related genomes, a cluster- or taxonomy-aware split is still preferable to reduce homology leakage between partitions.

No training genomes, checkpoints or generated biological datasets are distributed in this repository.

## License

No license has been assigned yet.
