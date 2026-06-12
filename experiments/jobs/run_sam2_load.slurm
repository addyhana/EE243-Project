#!/bin/bash
#SBATCH --job-name=sam2_load_test
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

${SAM2_PYTHON} \
experiments/scripts/test_sam2_load.py