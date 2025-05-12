#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=cost_quality
#SBATCH --output=cost_quality.log   
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6          #  På g001 bør en berense seg til 96 / 16 cores per GPU
#SBATCH --gres=gpu:1

srun -n 1 python cost-quality.py minatar-asterix 50 20 1