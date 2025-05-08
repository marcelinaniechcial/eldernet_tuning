#Set job requirements
#!/bin/bash
#SBATCH -J eldernet_tuning
#SBATCH -N 1    
#SBATCH -p gpu
#SBATCH -t 00:01:00
#SBATCH --gpus-per-node=1
#SBATCH -o logs/output.%j.out

#Loading modules

module load 2023
module load Anaconda3/2023.07-2
conda activate /projects/0/einf2658/users/mniechcial/envs/myenv


# #Set environment variables
# export PYTHONPATH=$PYTHONPATH:/projects/einf2658/users/TJ/ssl_gait
# PD_PATH=/projects/einf2658/users/TJ/ssl_gait/processed_data

# #Copy data to scratch
# echo "Copying data to scratch..."
# cp -r $PD_PATH/1.preprocessing $TMPDIR/1.preprocessing
# echo "Done copying!"

python -u training.py 
