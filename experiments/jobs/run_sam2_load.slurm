#!/bin/bash
#SBATCH --job-name=sam2_load_test
#SBATCH --output=/home/alamp009/EE243-Project/experiments/logs/sam2_%j.out
#SBATCH --error=/home/alamp009/EE243-Project/experiments/logs/sam2_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

mkdir -p /home/alamp009/EE243-Project/experiments/logs

cd /home/alamp009/EE243-Project
/home/alamp009/.conda/envs/sam2/bin/python \
experiments/scripts/test_sam2_load.py