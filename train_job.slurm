#!/bin/bash
#SBATCH --job-name=EN-train    # create a short name for your job
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=03:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ameyathete11@gmail.com
#SBATCH --output=logs/%x-geometric-1p5-so3-e150.log

module load python
source activate pytorch-gnn

echo "python equivariant-tracking/train.py --pt=1.5 --construction='geometric' --save-model"
python equivariant-tracking/train.py --pt=1.5 --construction='geometric' --save-model