#!/bin/bash
#SBATCH -J efficiency-job   # create a short name for your job
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A m3443
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=00:40:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ameyathete11@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/EQ-EN-efficiency-geometric-L1-1p5.log

module load python cudnn/8.2.0
conda activate torch-gpu

echo "Efficiency @ pT = 1p5"
python equivariant-tracking/efficiency.py 1.5
