import torch
import logging
import os
import sys

def get_gpu_backend():
    """Returns the available GPU backend ('cuda' or 'rocm') or raises RuntimeError"""
    if torch.cuda.is_available():
        # Check if we're using ROCm by looking for HIP in the PyTorch build info
        if torch.version.hip is not None:
            return 'rocm'
        return 'cuda'
    else:
        raise RuntimeError("No GPU support available. This script requires either NVIDIA CUDA or AMD ROCm GPUs.")

# Get GPU backend or fail fast
GPU_BACKEND = get_gpu_backend()
NVIDIA_AVAILABLE = GPU_BACKEND == 'cuda'
ROCM_AVAILABLE = GPU_BACKEND == 'rocm'
logging.info(f"{GPU_BACKEND.upper()} GPU support available")

# Import appropriate GPU monitoring tools
if NVIDIA_AVAILABLE:
    import pynvml
    pynvml.nvmlInit()
elif ROCM_AVAILABLE:
    rocm_path = os.getenv("ROCM_PATH")
    if rocm_path is None:
        raise RuntimeError("ROCM_PATH environment variable not set")
    sys.path.append(f"{rocm_path}/libexec/rocm_smi/")
    from rocm_smi import rocm_smi
    rocm_smi.initializeRsmi()


def get_gpu_info(device_index):
    """Get GPU information in a vendor-agnostic way"""
    if NVIDIA_AVAILABLE:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'name': name,
            'total_memory': memory.total,
            'used_memory': memory.used,
            'free_memory': memory.free
        }
    elif ROCM_AVAILABLE:
        device_info = rocm_smi.getGPUInfo(device_index)
        return {
            'name': device_info['name'],
            'total_memory': device_info['total_memory'] * (1024 ** 2),  # Convert to bytes
            'used_memory': device_info['used_memory'] * (1024 ** 2),
            'free_memory': device_info['free_memory'] * (1024 ** 2)
        }


def initialize_gpu(local_rank):
    """Initialize GPU in a vendor-agnostic way"""
    if torch.cuda.is_available():  # This works for both CUDA and ROCm
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if NVIDIA_AVAILABLE:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        elif ROCM_AVAILABLE:
            handle = local_rank  # ROCm SMI uses device index directly
        else:
            handle = None
    else:
        device = torch.device("cpu")
        handle = None
        
    return device, handle

def get_profiler():
    """Returns appropriate profiling tools based on GPU backend"""
    if NVIDIA_AVAILABLE:
        return torch.cuda.nvtx
    elif ROCM_AVAILABLE:
        return torch.profiler
    return None