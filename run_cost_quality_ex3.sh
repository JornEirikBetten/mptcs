#!/bin/bash

#SBATCH --account=jorneirik 
#SBATCH --job-name=cost_quality
#SBATCH --output=cost_quality.log   
#SBATCH --partition=dgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 6          #  På g001 bør en berense seg til 96 / 16 cores per GPU
#SBATCH --gres=gpu:1

srun -n 1 python cost-quality.py minatar-asterix 50 20 1
srun -n 1 python cost-quality.py minatar-asterix 50 20 2
srun -n 1 python cost-quality.py minatar-asterix 50 20 3
srun -n 1 python cost-quality.py minatar-asterix 50 20 4

srun -n 1 python cost-quality.py minatar-breakout 50 20 1
srun -n 1 python cost-quality.py minatar-breakout 50 20 2
srun -n 1 python cost-quality.py minatar-breakout 50 20 3
srun -n 1 python cost-quality.py minatar-breakout 50 20 4

srun -n 1 python cost-quality.py minatar-seaquest 50 20 1
srun -n 1 python cost-quality.py minatar-seaquest 50 20 2
srun -n 1 python cost-quality.py minatar-seaquest 50 20 3
srun -n 1 python cost-quality.py minatar-seaquest 50 20 4

srun -n 1 python cost-quality.py minatar-space_invaders 50 20 1
srun -n 1 python cost-quality.py minatar-space_invaders 50 20 2
srun -n 1 python cost-quality.py minatar-space_invaders 50 20 3
srun -n 1 python cost-quality.py minatar-space_invaders 50 20 4



