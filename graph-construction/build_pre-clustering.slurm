#!/bin/bash
#SBATCH -J build-graph-precluster    # create a short name for your job
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A m3443
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=20G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=02:00:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ameyathete11@gmail.com
#SBATCH --mail-type=ALL

module load python
conda activate pytorch-gnn

echo "...building pre-clustering"
python equivariant-tracking/graph-construction/build_pre-clustering.py equivariant-tracking/graph-construction/configs/pre-clustering.yaml --start-evtid=1000 --end-evtid=2500
echo "...done"
