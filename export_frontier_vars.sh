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

# export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HSA_ENABLE_SDMA=0