#!/bin/bash
#SBATCH --job-name=EN-train
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=03:30:00          # total run time limit (HH:MM:SS)

module load python
source activate torch-gpu

TZ='Asia/Kolkata' date
echo "python equivariant-tracking/train_so2.py --save-model"
python equivariant-tracking/train_so2.py --save-model