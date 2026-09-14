#!/bin/bash
#SBATCH --job-name=kmer_lm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
mkdir -p logs checkpoints

# Load the Python/PyTorch environment appropriate for your cluster before this step.
# Example:
# module load <your-pytorch-module>

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --data-dir data/processed \
  --output checkpoints/kmer_lm.pt \
  --epochs 10 \
  --batch-size 8
