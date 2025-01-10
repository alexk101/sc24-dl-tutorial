#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m4790_g
#SBATCH -q debug
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --signal=B:USR1@90  # Send signal 10 minutes before time limit
#SBATCH --time=00:30:00
#SBATCH --image=nersc/pytorch:24.06.02
#SBATCH --module=gpu,nccl-plugin
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out

# Handle SLURM signals
# These are used to handle the time limit and checkpointing
cleanup_handler() {
    echo "Received cleanup signal - terminating job"
    scancel $SLURM_JOB_ID
}
trap 'cleanup_handler' USR1

# Set up the data and log directories
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
# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0


set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp_mod.py ${args}
    "