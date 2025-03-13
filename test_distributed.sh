#!/bin/bash
#SBATCH -A geo163
#SBATCH -J dist_test
#SBATCH -o dist_test-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

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

# Add these critical environment variables for ROCm
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_ENABLE_SDMA=0

CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Run with explicit GPU binding
srun --ntasks=16 --ntasks-per-node=8 --gpus-per-node=8 ${CONDA_ENV_PATH}/bin/python test_distributed.py