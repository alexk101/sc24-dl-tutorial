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

# Handle SLURM signals
# These are used to handle the time limit and checkpointing
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

# Location of the conda environment
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Command line arguments
args="${@}"

# Run with srun directly - no bash -c wrapper
srun --ntasks=$((SLURM_NNODES*8)) --ntasks-per-node=8 --gpus-per-node=8 \
  ${CONDA_ENV_PATH}/bin/python train_mp_mod.py ${args}