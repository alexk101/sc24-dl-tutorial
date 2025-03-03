
Ideally, you want both because:

1. Python buffer handles normal cases
2. SLURM signal is backup if Python fails
3. Different time scales provide layered protection

Job Timeline:
[....Training....][Python Buffer (5min)][SLURM Signal (90s)][KILL]
                  ^                     ^                    ^
                  |                     |                    |
        Graceful checkpoint     Emergency cleanup     Force terminate


# On Node Configuration for Frontier



# Understanding Tensor Parallelism Across Nodes

You're right to question this point. Let me clarify how tensor parallelism works in this distributed training setup:

## Tensor Parallelism Configuration

When you set `--tensor_parallel=8` with 8 GPUs per node, you're creating tensor parallel groups of 8 GPUs each. The key issue is how these groups are formed across your multi-node setup.

Looking at your log, I can see:
```
2025-02-28 15:44:22,455 - root - INFO - Setting DP = 2, TP = 8, CP = 1, PP = 1
```

This means:
- Tensor Parallel (TP) = 8 GPUs
- Data Parallel (DP) = 2 groups
- Total GPUs = 16 (across 2 nodes)

## The Problem

The issue is that your tensor parallel groups are spanning across nodes. With 8 GPUs per node and TP=8, each tensor parallel group should ideally fit within a single node. However, the way the ranks are assigned in your implementation might be creating tensor parallel groups that cross node boundaries.

Looking at the NCCL channel information in your logs:
```
frontier08685:197357:197575 [0] NCCL INFO Channel 00/16 : 0 4 5 3 2 14 15 9 8 12 13 11 10 6 7 1
```

This suggests that the communication patterns are crossing node boundaries, which requires much higher latency inter-node communication.

## Solution

You should modify your approach to ensure tensor parallel groups stay within node boundaries:

1. **Verify rank assignment**: Check how your `distributed/comm.py` or equivalent is assigning ranks to tensor parallel groups. The ideal assignment would keep tensor parallel groups within a node.

2. **Modify the order parameter**: Your code likely has an `order` parameter that controls rank assignment. Make sure it's set to keep tensor parallel ranks closest:
   ```
   order=tp-dp-cp
   ```
   
   In your log, I see:
   ```
   2025-02-28 15:44:22,539 - root - INFO - order tp-cp-dp
   ```
   
   This looks correct, but the implementation might not be respecting node boundaries.

3. **Try a different configuration**: Instead of spanning TP across nodes, try:
   ```
   sbatch --nodes 2 submit_frontier.sh --config=mp --tensor_parallel=4 --scale_depth=12 --scale_heads=8 --scale_dim=384 --n_train=25 --local_batch_size=16 --num_data_workers=1
   ```
   
   This would create 4 tensor parallel groups (each with 4 GPUs) that stay within node boundaries, and use data parallelism across nodes.

The key insight is that tensor parallelism requires much more communication than data parallelism, so you want to keep tensor parallel groups within the same node to leverage the high-bandwidth intra-node connections.
