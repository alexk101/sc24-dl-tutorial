export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
# export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
# Try using the IP address directly instead of hostname
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1 | xargs getent hosts | awk '{print $1}')