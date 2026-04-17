#!/bin/bash

#SBATCH --job-name=time_sim_lazy
#SBATCH --output=/work/%u/%x-%j.out
#SBATCH --error=/work/%u/%x-%j.err
#SBATCH --time=01-12:00:00

#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1

ml load Mamba
conda activate unseen-awg

python ./analyses/scripts/time_simulation_lazy.py