import torch
import pynvml
import logging

def get_gpu_backend():
    """Returns the available GPU backend ('cuda' or 'rocm') or raises RuntimeError"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        return 'rocm'
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
    from rocm_smi import rocm_smi
    rocm_smi.initialize()


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
    # Set device and benchmark mode based on available backend
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if NVIDIA_AVAILABLE:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        else:
            handle = None
    elif hasattr(torch, 'hip') and torch.hip.is_available():
        torch.backends.hip.benchmark = True
        device = torch.device(f"hip:{local_rank}")
        if ROCM_AVAILABLE:
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