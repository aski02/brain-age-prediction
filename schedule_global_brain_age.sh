#!/bin/bash
#SBATCH --job-name=global_brain_age      # Specify job name
#SBATCH --partition=gpu        # Specify partition name
#SBATCH --nodes=1              # Specify number of nodes
#SBATCH --ntasks=1              # Specify number of processes (tasks) 
#SBATCH --cpus-per-task=32     # Specify number of CPUs (cores) per task
#SBATCH --mem-per-cpu=1G       # Specify amount of memory per CPU
#SBATCH --gres=gpu:1           # Generic resources; 1 GPU
#SBATCH --exclusive            # Use the node exclusively; recommended for jobs with high data traffic
#SBATCH --time=24:00:00        # Set a limit on the total run time
#SBATCH --mail-type=FAIL       # Notify user by email in case of job failure
#SBATCH --account=sc-users     # Charge resources on this project account
#SBATCH --output=my_job.o%j    # File name for standard output
#SBATCH --error=my_job.e%j     # File name for standard error output

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
date
source ~/.bashrc
conda activate env_training

echo "Conda Environment: $(conda info --envs | grep '*')"
echo "Python Version: $(python --version)"

python train.py --config_file configs/configs_whole_brain.yaml --gpu_id 0
