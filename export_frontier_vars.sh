# Add these environment variables before source export_DDP_vars.sh
# export NCCL_DEBUG=INFO # For debugging network issues
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_FAMILY=ipv4
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_CROSS_NIC=1       # On large systems, this NCCL setting has been found to improve perf
export PMI_NO_FORK=1
export NCCL_IB_HCA=hsn0