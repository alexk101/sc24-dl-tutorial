#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J param-sweep-python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 00:30:00
#SBATCH --signal=B:USR1@90  # Send signal 10 minutes before time limit
#SBATCH -o %x-%j.out
#SBATCH --ntasks-per-node=8  # Changed from 1 to 8 for MI250X GPUs
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest

# Handle SLURM signals
# These are used to handle the time limit and checkpointing
cleanup_handler() {
    echo "Received cleanup signal - terminating job"
    scancel $SLURM_JOB_ID
}
trap 'cleanup_handler' USR1

# Set up the data and log directories
# DATADIR=/pscratch/sd/a/akiefer/era5
DATADIR=/pscratch/sd/s/shas1693/data/sc24_tutorial_data
LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
cd $SLURM_SUBMIT_DIR

# Reversing order of GPUs to match default CPU affinities from Slurm
module load miniforge3/23.11.0-0
module load rocm/6.2.4

set -e

source export_DDP_vars.sh
source export_frontier_vars.sh
/ccs/home/kiefera/.conda/envs/pytorch/bin/python train_mp_mod.py ${args}
