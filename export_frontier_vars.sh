# Add these environment variables before source export_DDP_vars.sh
export NCCL_DEBUG=INFO # For debugging network issues
# export NCCL_DEBUG=WARN # For debugging network issues, less verbose
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4
# export MPICH_GPU_SUPPORT_ENABLED=1
# export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve perf
# export PMI_NO_FORK=1
# export NCCL_IB_HCA=hsn0

# # Alternative approach to isolate NCCL communications
# export TORCH_NCCL_BLOCKING_WAIT=1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_NTHREADS=1

# Use job-specific temporary directory for NCCL
export NCCL_TEMP_DIR="/tmp/nccl-${SLURM_JOB_ID}"
mkdir -p $NCCL_TEMP_DIR

export NCCL_DMABUF_ENABLE=1
export NCCL_NET_GDR_LEVEL=3

# # Ensure we use the correct network interface
# export NCCL_NET_GDR_READ=1

# export NCCL_LAUNCH_MODE=PARALLEL

# # Prevent NCCL from trying to connect to other jobs
# export NCCL_SINGLE_RING_THRESHOLD=0
# export NCCL_MIN_NCHANNELS=1
# export NCCL_TOPOLOGY=FLAT
