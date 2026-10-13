#!/bin/bash
#SBATCH --job-name=rate_test
#SBATCH --time=10:00:00
#SBATCH --mem=16G           
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=6   

#SBATCH --mail-type=BEGIN,END,FAIL    
#SBATCH --mail-user=ml19laaa@leeds.ac.uk  



#module load cuda
#nvidia-smi


export PATH=/mnt/scratch/ml19laaa/miniconda3/bin:$PATH

# Load Conda and Activate the Correct Environment
source activate env1


# Log in to WandB (for remote clusters, API key must be set manually)
export WANDB_API_KEY='ef357d47e05ee464a3101a93cc8df02d58e14e9c'


# Run Python Script


python __main__.py


#python Train_test.py


#python test.py