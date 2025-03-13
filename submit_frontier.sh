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
# These are used to handle the time limit and checkpointing
cleanup_handler() {
    echo "Received cleanup signal - terminating job"
    scancel $SLURM_JOB_ID
}
trap 'cleanup_handler' USR1

# Set up the data and log directories
# DATADIR=/pscratch/sd/a/akiefer/era5

# Print the ROCM version on each node
scontrol show hostnames $SLURM_NODELIST > job.node.list
input="./job.node.list"
readarray -t arr <"$input"

for row in "${arr[@]}";do
  row_array=(${row})
  first=${row_array[0]}
  echo ${first}
  cmd="ssh ${USER}@${first} /opt/rocm-6.2.4/bin/rocm-smi"
  echo $cmd
  $cmd
done

# Set up environment variables
export DATADIR=/lustre/orion/geo163/proj-shared/downsampled_data
export SCRATCH=/lustre/orion/geo163/scratch/kiefera
export MACHINE=frontier
export HDF5_USE_FILE_LOCKING=FALSE

# Location of the conda environment
CONDA_ENV_PATH=/ccs/home/kiefera/.conda/envs/pytorch

# Command line arguments
args="${@}"

# Run with srun, sourcing environment variables inside each task
set -x
srun --export=ALL \
    bash -c "
    # Set GPU visibility for this task
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    export HIP_VISIBLE_DEVICES=\$SLURM_LOCALID
    export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
    
    # Source DDP and Frontier-specific variables
    source export_DDP_vars.sh
    source export_frontier_vars.sh
    
    # Print debug info
    echo \"Task \$SLURM_PROCID: RANK=\$RANK, LOCAL_RANK=\$LOCAL_RANK, HIP_VISIBLE_DEVICES=\$HIP_VISIBLE_DEVICES\"
    
    # Run the Python script
    ${CONDA_ENV_PATH}/bin/python train_mp_mod.py ${args}
    "