import numpy as np
import torch
import time
from utils.logging_utils import GLOBAL_LOG

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

@torch.jit.script
def weighted_rmse_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def weighted_acc_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_channels(pred, target)
    return torch.mean(result, dim=0)

def time_communication(comm, device):
    """Time communication operations with proper error handling"""
    stats = {
        "all_reduce_time_ms": 0.0,
        "broadcast_time_ms": 0.0
    }
    
    if not comm or not torch.distributed.is_initialized():
        GLOBAL_LOG.error("Communication not initialized")
        return stats
    
    try:
        # Time all_reduce
        tensor = torch.ones(1, device=device)
        torch.cuda.synchronize()
        start = time.time()
        torch.distributed.all_reduce(tensor, group=comm.get_group("dp"))
        torch.cuda.synchronize()
        end = time.time()
        stats["all_reduce_time_ms"] = (end - start) * 1000
        
        # Time broadcast - use the first rank in the DP group as source
        tensor = torch.ones(1, device=device)
        torch.cuda.synchronize()
        start = time.time()
        
        # Get the first rank in the DP group to be the source
        dp_group = comm.get_group("dp")
        dp_ranks = comm.get_ranks_in_group("dp")
        src_rank = dp_ranks[0] if dp_ranks else 0
        
        torch.distributed.broadcast(tensor, src=src_rank, group=dp_group)
        torch.cuda.synchronize()
        end = time.time()
        stats["broadcast_time_ms"] = (end - start) * 1000
        
        return stats
    except Exception as e:
        GLOBAL_LOG.error(f"Error in time_communication: {e}")
        return stats

def backward_with_comm_timing(loss, optimizer):
    # Start timing
    torch.cuda.synchronize()
    backward_start = time.time()
    
    # Do backward pass
    loss.backward()
    
    # Synchronize before communication
    torch.cuda.synchronize()
    compute_time = time.time() - backward_start
    
    # Time the all-reduce operations (if using DDP)
    comm_start = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    comm_time = time.time() - comm_start
    
    return {
        "backward_compute_time": compute_time,
        "comm_time": comm_time,
        "comm_ratio": comm_time / (compute_time + comm_time)
    }