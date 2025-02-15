#!/bin/bash

scale_dim=512
nodes=4
time_limit=03:00:00
local_batch_size=8

# Arrays for parameter sweeps
patch_sizes=(4 8 16)
learning_rates=(1E-4 5E-4 1E-3)
n_train_years=("10" "15" "20" "25")

# Calculate compute hours at risk
total_hours=0
# Convert HH:MM:SS to hours (using awk to handle floating point)
hours=$(echo "$time_limit" | awk -F: '{ print ($1 + $2/60 + $3/3600) }')

# Calculate total compute hours for all combinations
for n_train_year in "${n_train_years[@]}"; do
    for patch_size in "${patch_sizes[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            compute_hours=$(awk "BEGIN {print $hours * $nodes}")
            total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
            echo "Time limit: ${time_limit}, Compute hours at risk: ${compute_hours}"
        done
    done
done

echo "Total compute hours at risk across all runs: ${total_hours}"

# Ask for confirmation
read -p "Do you want to proceed with submitting these jobs? y/[n] " confirm
if [[ $confirm != [yY] ]]; then
    echo "Job submission cancelled."
    exit 0
fi

# Submit jobs for all combinations
for n_train_year in "${n_train_years[@]}"; do
    for patch_size in "${patch_sizes[@]}"; do
        for learning_rate in "${learning_rates[@]}"; do
            sbatch --nodes ${nodes} submit_scaling.sh --config=mp --tensor_parallel=4 \
                    --scale_depth=12 --scale_heads=8 --scale_dim=${scale_dim} \
                    --n_train=${n_train_year} --local_batch_size=${local_batch_size} \
                    --time_limit=${time_limit} --n_nodes=${nodes} --amp_mode=none \
                    --patch_size=${patch_size} --learning_rate=${learning_rate}
        done
    done
done

