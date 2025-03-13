#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J param-sweep-python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8  # Changed from 1 to 8 for MI250X GPUs
#SBATCH --gpus-per-task=1

# Handle SLURM signals
cleanup_handler() {
    echo "Received cleanup signal - terminating job"
    scancel $SLURM_JOB_ID
}
trap 'cleanup_handler' USR1

# Load modules first - explicitly
module purge
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/1.22.0

# Set up environment variables
export DATADIR=/lustre/orion/geo163/proj-shared/downsampled_data
export SCRATCH=/lustre/orion/geo163/scratch/kiefera
LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}
export MACHINE=frontier
export HDF5_USE_FILE_LOCKING=FALSE

# Source environment variables for DDP - before srun
source export_DDP_vars.sh
source export_frontier_vars.sh

# Set master address
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=3442

# Location of the conda environment
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Command line arguments
args="${@}"

# Create a modified test script that uses MPI ranks for GPU assignment
cat > test_gpu_mpi.py << 'EOF'
import os
import sys
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
node_size = 8  # Number of GPUs per node on Frontier
local_rank = rank % node_size

# Set GPU visibility based on MPI local rank
os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
os.environ["HIP_VISIBLE_DEVICES"] = str(local_rank)
os.environ["ROCR_VISIBLE_DEVICES"] = str(local_rank)

# Set PyTorch distributed environment variables
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["LOCAL_RANK"] = str(local_rank)

# Print environment variables
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
print(f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
print(f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}")
print(f"RANK={os.environ.get('RANK')}")
print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
print(f"MPI Rank={rank}, World Size={world_size}")

# Now import torch
import torch
print(f"PyTorch version: {torch.__version__}")
if hasattr(torch.version, 'hip'):
    print(f"PyTorch HIP version: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Try to create a tensor on GPU
try:
    x = torch.ones(1, device='cuda')
    print(f"Successfully created tensor on GPU: {x}")
except Exception as e:
    print(f"Failed to create tensor on GPU: {e}")
EOF

# Run with srun directly - no bash -c wrapper
srun --ntasks=8 --ntasks-per-node=8 --gpus-per-node=8 --gpus-per-task=1 ${CONDA_ENV_PATH}/bin/python test_gpu_mpi.py