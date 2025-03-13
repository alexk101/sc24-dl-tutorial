import os
import sys
from mpi4py import MPI  # Import MPI first

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

# Print environment variables
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}")
print(f"ROCR_VISIBLE_DEVICES={os.environ.get('ROCR_VISIBLE_DEVICES')}")
print(f"SLURM_LOCALID={os.environ.get('SLURM_LOCALID')}")
print(f"RANK={os.environ.get('RANK')}")
print(f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
print(f"MPI Rank={rank}, World Size={world_size}")

# Now import torch
import torch
print(f"PyTorch version: {torch.__version__}")
if hasattr(torch.version, 'hip'):
    print(f"PyTorch HIP version: {torch.version.hip}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

# Try to create a tensor on GPU
try:
    x = torch.ones(1, device='cuda')
    print(f"Successfully created tensor on GPU: {x}")
except Exception as e:
    print(f"Failed to create tensor on GPU: {e}")