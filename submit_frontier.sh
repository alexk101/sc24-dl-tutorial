#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J param-sweep-python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8  # Changed from 1 to 8 for MI250X GPUs
#SBATCH --cpus-per-task=7
#SBATCH --gpu-bind=closest

ROCM_VERSION=6.2.4

[ -z $JOBID ] && JOBID=$SLURM_JOB_ID
[ -z $JOBSIZE ] && JOBSIZE=$SLURM_JOB_NUM_NODES

module purge
module load PrgEnv-gnu/8.5.0
module load rocm/${ROCM_VERSION}
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
export LD_LIBRARY_PATH=/ccs/home/kiefera/scratch/rccl/aws-ofi-rccl/lib:/opt/rocm-${ROCM_VERSION}/lib:$LD_LIBRARY_PATH

# MIOpen
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/$JOBID
mkdir -p $MIOPEN_USER_DB_PATH

LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}

export OMP_NUM_THREADS=7
export HDF5_USE_FILE_LOCKING=FALSE
cd $SLURM_SUBMIT_DIR

# Location of the conda environment
CONDA_BASE=/sw/frontier/miniforge3/23.11.0-0
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Store the arguments before activating conda
PYTHON_ARGS=("$@")

set -x

source export_DDP_vars.sh
source export_frontier_vars.sh
export MASTER_PORT=3442 # default from torch launcher

# Run the command with proper argument handling
srun -n $((SLURM_JOB_NUM_NODES*8)) ${CONDA_ENV_PATH}/bin/python train_mp_mod.py "${PYTHON_ARGS[@]}" --checkpoint_freq 100 --num_data_workers ${OMP_NUM_THREADS}