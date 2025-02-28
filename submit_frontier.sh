#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J param-sweep-python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out
#SBATCH --gpus-per-node 8
#SBATCH --ntasks-per-node=8  # Changed from 1 to 8 for MI250X GPUs
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

module purge
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/1.22.0

# Handle SLURM signals
# These are used to handle the time limit and checkpointing
cleanup_handler() {
    echo "Received cleanup signal - terminating job"
    scancel $SLURM_JOB_ID
}
trap 'cleanup_handler' USR1

# Set up the data and log directories
# DATADIR=/pscratch/sd/a/akiefer/era5
export DATADIR=/lustre/orion/geo163/proj-shared/downsampled_data
export SCRATCH=/lustre/orion/geo163/scratch/kiefera
export MACHINE=frontier

# RCCL
export LD_LIBRARY_PATH=/ccs/home/kiefera/scratch/rccl/aws-ofi-rccl/lib:$LD_LIBRARY_PATH

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE
cd $SLURM_SUBMIT_DIR

# Location of the conda environment
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch
source activate ${CONDA_ENV_PATH}

set -x

source export_DDP_vars.sh
source export_frontier_vars.sh
export MASTER_PORT=3442 # default from torch launcher
srun ${CONDA_ENV_PATH}/bin/python train_mp_mod.py ${args}
