# Add these environment variables before source export_DDP_vars.sh
# export NCCL_DEBUG=INFO # For debugging network issues
# export NCCL_DEBUG=WARN # For debugging network issues, less verbose
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve perf
export NCCL_IB_HCA=hsn0

# Use job-specific temporary directory for NCCL
export NCCL_TEMP_DIR="/tmp/nccl-${SLURM_JOB_ID}"
mkdir -p $NCCL_TEMP_DIR

# Ensure we use the correct network interface
export NCCL_NET_GDR_LEVEL=3
export NCCL_PROTO=Simple