#!/bin/bash
#SBATCH --job-name=sam2_exp3
#SBATCH --output=experiments/logs/sam2_%j.out
#SBATCH --error=experiments/logs/sam2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -a
source .env
set +a

mkdir -p ${PROJECT_ROOT}/experiments/logs

mkdir -p results/experiment3/masks
mkdir -p results/experiment3/overlays

rm -f results/experiment3/masks/*
rm -f results/experiment3/overlays/*

${SAM2_PYTHON} \
experiments/scripts/test_sam2_baseline.py \
  --video_dir experiments/input_frames/experiment3 \
  --first_mask experiments/input_first_mask/experiment3/first_mask.png \
  --output_dir results/experiment3