#!/bin/bash
#SBATCH -A geo163
#SBATCH -J dist_test
#SBATCH -o dist_test-%j.out
#SBATCH -t 00:10:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Node list: $SLURM_NODELIST"
echo "========================"

# Unset any global GPU visibility variables to avoid conflicts
unset CUDA_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

# Load modules
module purge
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/1.22.0

# Print loaded modules
echo "=== Loaded Modules ==="
module list
echo "======================"

# Source environment variables for DDP
source export_DDP_vars.sh
source export_frontier_vars.sh

# Override MASTER_PORT if needed
export MASTER_PORT=3442

# Print DDP environment variables
echo "=== DDP Environment Variables ==="
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "================================="

# Print library paths
echo "=== Library Paths ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "===================="

CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Print Python and PyTorch information
echo "=== Python Information ==="
$CONDA_ENV_PATH/bin/python -c "import sys; print(f'Python version: {sys.version}')"
$CONDA_ENV_PATH/bin/python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
echo "=========================="

# Run with explicit GPU binding
echo "=== Starting distributed test ==="
srun --ntasks=$((SLURM_NNODES*8)) --ntasks-per-node=8 --gpus-per-node=8 ${CONDA_ENV_PATH}/bin/python test_distributed.py
echo "=== Test completed ==="