#!/bin/bash
#SBATCH -J eldernet_tuning
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=1
#SBATCH -o logs/output.%j.out
#SBATCH --tasks-per-node 1

#Loading modules

module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0


python -u training.py 
