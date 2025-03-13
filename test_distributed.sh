#!/bin/bash
#SBATCH -A YourAccount
#SBATCH -J dist_test
#SBATCH -o dist_test-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2  # Start with 2 nodes for testing

module purge
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/1.22.0

# Source environment variables for DDP
source export_DDP_vars.sh
source export_frontier_vars.sh

# Override MASTER_PORT if needed
export MASTER_PORT=3442

CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Run the test script
srun ${CONDA_ENV_PATH}/bin/python test_distributed.py