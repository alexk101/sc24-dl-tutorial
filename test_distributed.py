#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.distributed as dist
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Rank %(rank)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"dist_test_rank{os.environ.get('RANK', '?')}.log")
    ]
)

# Custom logger that includes rank
class RankLogger:
    def __init__(self, rank):
        self.rank = rank
        self.logger = logging.getLogger()
        
    def info(self, msg):
        self.logger.info(msg, extra={'rank': self.rank})
        
    def error(self, msg):
        self.logger.error(msg, extra={'rank': self.rank})

def main():
    # Get environment variables
    try:
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        master_addr = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        sys.exit(1)
    
    # Create logger with rank
    log = RankLogger(rank)
    
    # Log environment variables
    log.info(f"Environment: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    log.info(f"Environment: MASTER_ADDR={master_addr}, MASTER_PORT={master_port}")
    log.info(f"Node: {os.environ.get('SLURMD_NODENAME')}, "
        f"SLURM_PROCID: {os.environ.get('SLURM_PROCID')}, "
        f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID')}, "
        f"SLURM_NODEID: {os.environ.get('SLURM_NODEID')}")

    # Initialize process group
    log.info("Initializing process group")
    try:
        dist.init_process_group(
            backend="nccl",
            # init_method=f"tcp://{master_addr}:{master_port}",
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        log.info("Process group initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize process group: {e}")
        sys.exit(1)
    
    # Get device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        log.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")
    
    # Test all_reduce
    log.info("Testing all_reduce operation")
    try:
        # Create a tensor with value = rank
        tensor = torch.tensor([float(rank)], device=device)
        log.info(f"Initial tensor value: {tensor.item()}")
        
        # Perform all_reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = (world_size * (world_size - 1)) // 2  # Sum of 0 to (world_size-1)
        log.info(f"After all_reduce: {tensor.item()} (expected: {expected_sum})")
        
        # Verify result
        if abs(tensor.item() - expected_sum) < 1e-5:
            log.info("All_reduce test PASSED")
        else:
            log.error(f"All_reduce test FAILED: got {tensor.item()}, expected {expected_sum}")
    except Exception as e:
        log.error(f"All_reduce test FAILED with exception: {e}")
    
    # Test broadcast
    log.info("Testing broadcast operation")
    try:
        # Create a tensor with different values on each rank
        tensor = torch.tensor([float(rank * 10)], device=device)
        original_value = tensor.item()
        log.info(f"Before broadcast: {original_value}")
        
        # Broadcast from rank 0
        dist.broadcast(tensor, src=0)
        log.info(f"After broadcast: {tensor.item()} (expected: 0.0)")
        
        # Verify result
        if rank == 0:
            if abs(tensor.item() - original_value) < 1e-5:
                log.info("Broadcast test PASSED for sender")
            else:
                log.error(f"Broadcast test FAILED for sender: value changed unexpectedly")
        else:
            if abs(tensor.item() - 0.0) < 1e-5:
                log.info("Broadcast test PASSED for receiver")
            else:
                log.error(f"Broadcast test FAILED for receiver: got {tensor.item()}, expected 0.0")
    except Exception as e:
        log.error(f"Broadcast test FAILED with exception: {e}")
    
    # Test all_gather
    log.info("Testing all_gather operation")
    try:
        # Create a tensor with value = rank
        tensor = torch.tensor([float(rank)], device=device)
        
        # Prepare output list
        gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
        
        # Perform all_gather
        dist.all_gather(gathered, tensor)
        
        # Convert to list for easier logging
        gathered_values = [t.item() for t in gathered]
        expected_values = list(range(world_size))
        log.info(f"All_gather results: {gathered_values}")
        
        # Verify result
        if gathered_values == expected_values:
            log.info("All_gather test PASSED")
        else:
            log.error(f"All_gather test FAILED: got {gathered_values}, expected {expected_values}")
    except Exception as e:
        log.error(f"All_gather test FAILED with exception: {e}")
    
    # Test barrier
    log.info("Testing barrier operation")
    try:
        # Sleep based on rank to create timing differences
        sleep_time = (world_size - rank) * 0.5
        log.info(f"Sleeping for {sleep_time} seconds before barrier")
        time.sleep(sleep_time)
        
        # Record time before barrier
        start_time = time.time()
        
        # Call barrier
        dist.barrier()
        
        # Record time after barrier
        end_time = time.time()
        log.info(f"Passed barrier after {end_time - start_time:.3f} seconds")
        log.info("Barrier test PASSED")
    except Exception as e:
        log.error(f"Barrier test FAILED with exception: {e}")
    
    # Clean up
    log.info("Cleaning up process group")
    try:
        dist.destroy_process_group()
        log.info("Process group destroyed successfully")
    except Exception as e:
        log.error(f"Failed to destroy process group: {e}")
    
    log.info("Test completed")

if __name__ == "__main__":
    main()
