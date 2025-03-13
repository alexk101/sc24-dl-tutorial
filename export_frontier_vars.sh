module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
module load cray-hdf5-parallel/1.12.2.11
module load libfabric/1.22.0
ENABLE_AWS_OFI_RCCL_PLUGIN=1
# RCCL
if [ "$ENABLE_AWS_OFI_RCCL_PLUGIN" -eq 1 ]; then
    export LD_LIBRARY_PATH=/ccs/home/kiefera/scratch/rccl/aws-ofi-rccl/lib:$LD_LIBRARY_PATH
fi
export MIOPEN_DISABLE_CACHE=1
USER=kiefera
export DATADIR=/lustre/orion/geo163/proj-shared/downsampled_data
export SCRATCH=/lustre/orion/geo163/scratch/kiefera
export MACHINE=frontier

# Add these environment variables before source export_DDP_vars.sh
# export NCCL_DEBUG=INFO # For debugging network issues
# export NCCL_DEBUG=WARN # For debugging network issues, less verbose
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4

# Use job-specific temporary directory for NCCL
export NCCL_TEMP_DIR="/tmp/nccl-${SLURM_JOB_ID}"
mkdir -p $NCCL_TEMP_DIR

export NCCL_DMABUF_ENABLE=1
export NCCL_NET_GDR_LEVEL=3
export NCCL_ALGO=Tree

export MASTER_PORT=3442 # default from torch launcher

export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_ENABLE_SDMA=0

