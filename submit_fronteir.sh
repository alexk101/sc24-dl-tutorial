#!/bin/bash
#SBATCH -A GEO163 
#SBATCH -J python
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -t 0:05:00

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

# scale_depth, scale_heads, scale_dim, job_name
# Base model size from config
BASE_DEPTH=12
BASE_HEADS=8
# 384, 576, 768, 1024
BASE_DIM=384
VALID_YEARS=1

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
cd $SLURM_SUBMIT_DIR

# Reversing order of GPUs to match default CPU affinities from Slurm
module load miniforge3/23.11.0-0
source activate my_env

set -e

source export_DDP_vars.sh
python train_mp_mod.py ${args}
