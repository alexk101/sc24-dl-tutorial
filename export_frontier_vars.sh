# Add these environment variables before source export_DDP_vars.sh
# export NCCL_DEBUG=INFO # For debugging network issues
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve perf
export PMI_NO_FORK=1
export NCCL_IB_HCA=hsn0

# Add these to isolate NCCL communication between jobs
export NCCL_COMM_ID="${SLURM_JOB_ID:-$(date +%s)}"  # Use SLURM job ID or timestamp
export NCCL_SOCKET_NTHREADS=1   # Reduce connection threads
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=LOC
export NCCL_NET_GDR_LEVEL=LOC
export NCCL_LAUNCH_MODE=PARALLEL  # Helps with multiple jobs
