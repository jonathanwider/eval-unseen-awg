#!/bin/bash

#SBATCH --job-name=time_sim
#SBATCH --output=/work/%u/%x-%j.out
#SBATCH --error=/work/%u/%x-%j.err
#SBATCH --time=00-06:00:00

#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1

ml load Mamba
conda activate unseen-awg

python ./analyses/scripts/time_simulation.py