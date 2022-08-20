#!/bin/bash
#SBATCH --job-name=EN-train    # create a short name for your job
#SBATCH -A m3443_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ameyathete11@gmail.com

module load python
source activate pytorch-gnn

echo "python equivariant-tracking/train.py --pt=$1 --construction='geometric' --epochs=100 --lr=1e-3 --save-model"
python equivariant-tracking/train.py --pt=$1 --construction='geometric' --epochs=100 --lr=1e-3 --save-model
