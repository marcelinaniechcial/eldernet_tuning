#!/bin/bash
#SBATCH -J eldernet_tuning_eval
#SBATCH -N 1
#SBATCH -p gpu_a100
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=1
#SBATCH -o logs/output_eval.%j.out

#Loading modules
module purge
module load 2024
module load Python/3.10.12-GCCcore-13.2.0

python -u evaluations.py 
