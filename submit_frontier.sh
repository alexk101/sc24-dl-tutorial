#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J param-sweep-python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:05:00
#SBATCH --signal=B:USR1@60  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out
#SBATCH --ntasks-per-node=8  # Changed from 1 to 8 for MI250X GPUs
#SBATCH --ntasks=$(($8 * $SLURM_NNODES))
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

unset SLURM_EXPORT_ENV

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
LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname -i)
cd $SLURM_SUBMIT_DIR

module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Location of the conda environment
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch
source activate ${CONDA_ENV_PATH}

set -e

source export_DDP_vars.sh
source export_frontier_vars.sh
${CONDA_ENV_PATH}/bin/python train_mp_mod.py ${args}
