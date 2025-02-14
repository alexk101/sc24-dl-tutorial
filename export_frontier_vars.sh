# Add these environment variables before source export_DDP_vars.sh
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_NET=UCX
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_REG_METHODS=direct
export UCX_TLS=rc,sm,self
export NCCL_DEBUG=INFO