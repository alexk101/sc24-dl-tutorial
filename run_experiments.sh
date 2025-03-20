#!/bin/bash

# Base configuration
BASE_CONFIG="mp"
BASE_NODES=128
TIME_LIMIT="06:00:00"
BASE_BATCH_SIZE=8
TIME_LIMIT_2="02:00:00"

# Arrays for parameter sweeps
SCALE_FACTORS=(1 2 4 8)
EMBED_DIMS=(128 256 512 1024)
DT_VALUES=(1 2 4 8)
TRAIN_YEARS=(1 5 10 15 20 25)
NODE_COUNTS=(1 2 4 8 16)
AMP_MODES=("none" "fp16" "bf16")
BATCH_SIZES=(1 2 4 8 16 32 64)
PATCH_SIZES=(2 4 8 16)

# Calculate compute hours at risk
total_hours=0
# Convert HH:MM:SS to hours (using awk to handle floating point)
hours=$(echo "$TIME_LIMIT" | awk -F: '{ print ($1 + $2/60 + $3/3600) }')

# Calculate total compute hours for all combinations
echo "Calculating total compute hours..."

# Model Scaling Experiments
for scale in "${SCALE_FACTORS[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

for dim in "${EMBED_DIMS[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

for dt in "${DT_VALUES[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

# Patch Size Scaling
for patch_size in "${PATCH_SIZES[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

# Data Scaling Experiments
for years in "${TRAIN_YEARS[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

# Compute Resource Scaling Experiments
for nodes in "${NODE_COUNTS[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $nodes}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

for amp_mode in "${AMP_MODES[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

# Training & Inference Efficiency Experiments
for amp_mode in "${AMP_MODES[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

for batch_size in "${BATCH_SIZES[@]}"; do
    compute_hours=$(awk "BEGIN {print $hours * $BASE_NODES}")
    total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
done

echo "Total compute hours at risk across all runs: ${total_hours}"

# Ask for confirmation
read -p "Do you want to proceed with submitting these jobs? y/[n] " confirm
if [[ $confirm != [yY] ]]; then
    echo "Job submission cancelled."
    exit 0
fi

# 1. Model Scaling Experiments
echo "Running Model Scaling Experiments..."

# Parameter Count Scaling - Depth
echo "Running depth scaling experiments..."
for scale in "${SCALE_FACTORS[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="depth_scaling_${scale}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=$((12 * scale)) \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --exp_name="depth_scaling"
    
    rm "${temp_script}"
done

# Parameter Count Scaling - Heads
echo "Running attention heads scaling experiments..."
for scale in "${SCALE_FACTORS[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="heads_scaling_${scale}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=$((8 * scale)) \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --exp_name="heads_scaling"

    rm "${temp_script}"
done

# Embedding Size Scaling
for dim in "${EMBED_DIMS[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="embedding_scaling_${dim}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=${dim} \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --exp_name="embedding_scaling"
    
    rm "${temp_script}"
done

# Sequence Length Scaling
for dt in "${DT_VALUES[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="dt_scaling_${dt}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --dt=${dt} \
        --exp_name="dt_scaling"
    
    rm "${temp_script}"
done

# Patch Size Scaling
echo "Running Patch Size Scaling Experiments..."
for patch_size in "${PATCH_SIZES[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="patch_size_scaling_${patch_size}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --patch_size=${patch_size} \
        --exp_name="patch_size_scaling"
    
    rm "${temp_script}"
done

# 2. Data Scaling Experiments
echo "Running Data Scaling Experiments..."

# Dataset Size Scaling
for years in "${TRAIN_YEARS[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="data_scaling_${years}"

    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=${years} \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --exp_name="data_scaling"
    
    rm "${temp_script}"
done

# 3. Compute Resource Scaling Experiments
echo "Running Compute Resource Scaling Experiments..."

# GPU Count Scaling
for nodes in "${NODE_COUNTS[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT_2}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="gpu_count_scaling_${nodes}"
    
    sbatch --nodes ${nodes} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${nodes} \
        --exp_name="gpu_count_scaling"
    
    rm "${temp_script}"
done

# Memory Optimization Experiments
for amp_mode in "${AMP_MODES[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="amp_mode_scaling_gc_${amp_mode}"

    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --amp_mode=${amp_mode} \
        --gradient_checkpointing \
        --exp_name="amp_mode_scaling_gc"
    
    rm "${temp_script}"
done

# 4. Training & Inference Efficiency Experiments
echo "Running Training & Inference Efficiency Experiments..."

# Precision Scaling
for amp_mode in "${AMP_MODES[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="amp_mode_scaling_${amp_mode}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${BASE_BATCH_SIZE} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --amp_mode=${amp_mode} \
        --exp_name="amp_mode_scaling"
    
    rm "${temp_script}"
done

# Batch Size Scaling
for batch_size in "${BATCH_SIZES[@]}"; do
    temp_script="submit_frontier_${RANDOM}.sh"
    sed "s/#SBATCH -t 00:30:00/#SBATCH -t ${TIME_LIMIT}/" submit_frontier.sh > "${temp_script}"
    export EXP_NAME="batch_size_scaling_${batch_size}"
    
    sbatch --nodes ${BASE_NODES} "${temp_script}" \
        --config=${BASE_CONFIG} \
        --tensor_parallel=4 \
        --scale_depth=12 \
        --scale_heads=8 \
        --scale_dim=384 \
        --n_train=25 \
        --local_batch_size=${batch_size} \
        --num_data_workers=1 \
        --n_nodes=${BASE_NODES} \
        --exp_name="batch_size_scaling"
    
    rm "${temp_script}"
done

echo "All experiments submitted!" 