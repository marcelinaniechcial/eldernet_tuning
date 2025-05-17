#!/bin/bash
#SBATCH -J eldernet_tuning_eval
#SBATCH -N 1    
#SBATCH -p gpu_a100
#SBATCH -t 10:00:00
#SBATCH --gpus-per-node=1
#SBATCH -o logs/output_eval.%j.out

#Loading modules

module load 2023
module load Anaconda3/2023.07-2
conda activate /projects/0/einf2658/users/mniechcial/envs/myenv

python -u evaluation.py 
