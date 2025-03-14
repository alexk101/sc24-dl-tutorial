export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

# Get the first node's hostname from SLURM_NODELIST
export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
echo "Setting MASTER_ADDR to $MASTER_ADDR"

# For debugging
echo "RANK=$RANK, LOCAL_RANK=$LOCAL_RANK, WORLD_SIZE=$WORLD_SIZE"