#!/bin/bash
#SBATCH --job-name=brain-age
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=24:00:00

python src/train_normal.py 0