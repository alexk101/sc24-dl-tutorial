export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS
export MASTER_PORT=29500 # default from torch launcher
# export MASTER_ADDR=$(hostname -i)
# export MASTER_ADDR=$SLURM_SUBMIT_HOST
export MASTER_ADDR=$HOSTNAME
