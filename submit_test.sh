#!/bin/bash

scale_dim=512
nodes=4
# Array of time limits in HH:MM:SS format
# time_limits=("00:30:00" "01:00:00" "01:30:00" "02:00:00" "02:30:00" "03:00:00")
time_limits=("00:30:00")

# Array of number of years to train on
# n_train_years=("10" "15" "20" "25")
n_train_years=("10")

# Calculate compute hours at risk
total_hours=0
for time_limit in "${time_limits[@]}"; do
    # Convert HH:MM:SS to hours (using awk to handle floating point)
    hours=$(echo "$time_limit" | awk -F: '{ print ($1 + $2/60 + $3/3600) }')
    
    for n_train_year in "${n_train_years[@]}"; do
        compute_hours=$(awk "BEGIN {print $hours * $nodes}")
        total_hours=$(awk "BEGIN {print $total_hours + $compute_hours}")
        echo "Time limit: ${time_limit}, Compute hours at risk: ${compute_hours}"
    done
done

echo "Total compute hours at risk across all runs: ${total_hours}"


for time_limit in "${time_limits[@]}"; do
  # Use sed to modify the time limit in a temporary copy of the submission script
  sed "s/#SBATCH --time=[0-9:]\+/#SBATCH --time=${time_limit}/" submit_scaling.sh > temp_submit.sh
  chmod +x temp_submit.sh
  
  # Submit the job with the modified script
  for n_train_year in "${n_train_years[@]}"; do
    sbatch --nodes ${nodes} temp_submit.sh --config=mp --tensor_parallel=4 \
            --scale_depth=12 --scale_heads=8 --scale_dim=${scale_dim} \
            --n_train=${n_train_year} --local_batch_size=4 --time_limit=${time_limit} \
            --n_nodes=${nodes}
  done
  
  # Clean up temporary script
  rm temp_submit.sh
done
