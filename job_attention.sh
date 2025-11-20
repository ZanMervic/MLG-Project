#!/bin/bash
#SBATCH --job-name=mlg-custom_attention
#SBATCH --output=logs/mlg_%j.out
#SBATCH --error=logs/mlg_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G

# Load your conda function
source $(conda info --base)/etc/profile.d/conda.sh

# Go to your code directory
cd /d/hpc/projects/FRI/zm3587/mlg

# Activate the env
conda activate project-mlg

# Finally launch
python hyperparameter_search_attention.py