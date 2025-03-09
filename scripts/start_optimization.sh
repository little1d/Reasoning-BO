#!/bin/sh


#SBATCH --job-name=bo  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --partition=AI4Chem

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bo

python /mnt/hwfile/ai4chem/yangzhuo/Faithful-BO/playground/pipeline/optimization_loop.py