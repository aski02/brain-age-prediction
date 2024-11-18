#!/bin/bash
#SBATCH --job-name=brain-age-neuro      # Specify job name
#SBATCH --partition=pgpu                # Specify partition name
#SBATCH --nodes=1                       # Specify number of nodes
#SBATCH --ntasks=8                      # Specify number of processes (tasks)
#SBATCH --cpus-per-task=32              # Specify number of CPUs (cores) per task
#SBATCH --gres=gpu:8                    # Request all 8 GPUs on the node
#SBATCH --exclusive                     # Use the node exclusively; recommended for jobs with high data traffic
#SBATCH --time=24:00:00                 # Increase the limit on the total run time to 24 hours
#SBATCH --mail-type=FAIL                # Notify user by email in case of job failure
#SBATCH --account=sc-users              # Charge resources on this project account
#SBATCH --output=brain_age_neuro.o%j    # File name for standard output
#SBATCH --error=brain_age_neuro.e%j     # File name for standard error output

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
date
source ~/.bashrc
conda activate env_training

echo "Conda Environment: $(conda info --envs | grep '*')"
echo "Python Version: $(python --version)"

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 0 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 1 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 2 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 3 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 4 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 5 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 6 &

srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
    python train.py --config_file configs/configs_neuromorphometrics.yaml --gpu_id 7 &

wait
