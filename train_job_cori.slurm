#!/bin/bash
#SBATCH --job-name=EN-train
#SBATCH --qos=regular
#SBATCH --time=03:30:00
#SBATCH --constraint=haswell
#SBATCH -c 10
#SBATCH -A m3443
#SBATCH --output=logs/so3/%x-geometric-1p5-e150.log

module load python
source activate pytorch-gnn

echo "python equivariant-tracking/train.py --pt=1.5 --construction='geometric' --save-model"
srun python equivariant-tracking/train.py --pt=1.5 --construction='geometric' --save-model