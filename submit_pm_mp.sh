#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m4790_g
#SBATCH -q debug
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=00:30:00
#SBATCH --image=nersc/pytorch:24.06.02
#SBATCH --module=gpu,nccl-plugin
#SBATCH --reservation=sc24_dl_tutorial_2
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/s/shas1693/data/sc24_tutorial_data
LOGDIR=${SCRATCH}/sc24-dl-tutorial/logs
mkdir -p ${LOGDIR}
args="${@}"

export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --cuda-graph-trace=node --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

# if cuda graphs, use train_mp_graphs.py
set -x
srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "
