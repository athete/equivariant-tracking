#!/bin/bash
#SBATCH -J build-graph    # create a short name for your job
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -A m3443
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=01:00:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-user=ameyathete11@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --output=logs/hitgraph-geometric-0p8.log

module load python
conda activate torch-gpu

echo "...building geometric"
python equivariant-tracking/graph-construction/build_geometric.py equivariant-tracking/graph-construction/configs/geometric.yaml --start-evtid=1000 --end-evtid=2500
echo "...done"
