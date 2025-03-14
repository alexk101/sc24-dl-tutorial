import torch
import logging
import os
import sys
import subprocess

NVIDIA_AVAILABLE = False
ROCM_AVAILABLE = False
GPU_BACKEND = None

def get_gpu_backend():
    """Returns the available GPU backend ('cuda' or 'rocm') or raises RuntimeError"""
    # Log environment variables for debugging
    if os.environ.get("MACHINE") == "frontier":
        hip_devices = os.environ.get("HIP_VISIBLE_DEVICES")
        rocr_devices = os.environ.get("ROCR_VISIBLE_DEVICES")
        slurm_localid = os.environ.get("SLURM_LOCALID")
        logging.info(f"HIP_VISIBLE_DEVICES={hip_devices}, ROCR_VISIBLE_DEVICES={rocr_devices}, SLURM_LOCALID={slurm_localid}")
    
    # Let PyTorch be the source of truth
    if torch.cuda.is_available():
        # Check if we're using ROCm by looking for HIP in the PyTorch build info
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return 'rocm'
        return 'cuda'
    else:
        raise RuntimeError("No GPU support available. This script requires either NVIDIA CUDA or AMD ROCm GPUs.")

def initialize_gpu_backend():
    global NVIDIA_AVAILABLE, ROCM_AVAILABLE, GPU_BACKEND
    GPU_BACKEND = get_gpu_backend()
    NVIDIA_AVAILABLE = GPU_BACKEND == 'cuda'
    ROCM_AVAILABLE = GPU_BACKEND == 'rocm'
    logging.info(f"{GPU_BACKEND.upper()} GPU support available")


def log_rocm_utilization():
    """Log ROCM utilization"""
    rocm_smi = os.environ['ROCM_PATH'] + '/bin/rocm-smi'
    result = subprocess.run([rocm_smi], capture_output=True, text=True)
    if result.returncode == 0:
        utilization_info = result.stdout
        logging.info(f"ROCM Utilization: {utilization_info}")


def get_gpu_info(device_index):
    """Get GPU information in a vendor-agnostic way"""
    # Import appropriate GPU monitoring tools
    if NVIDIA_AVAILABLE:
        import pynvml
        pynvml.nvmlInit()
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
        rocm_path = os.getenv("ROCM_PATH")
        if rocm_path is None:
            raise RuntimeError("ROCM_PATH environment variable not set")
        sys.path.append(f"{rocm_path}/libexec/rocm_smi/")
        import rocm_smi
        rocm_smi.initializeRsmi()
        (mem_used, mem_total) = rocm_smi.getMemInfo(device_index, "vram")
        return {
            'name': f"AMD GPU {device_index}",
            'total_memory': mem_total,  # Convert to bytes
            'used_memory': mem_used,
            'free_memory': (mem_total - mem_used)
        }


def initialize_gpu(local_rank):
    """Initialize GPU in a vendor-agnostic way"""
    # Check visible devices
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "")
    slurm_localid = os.environ.get("SLURM_LOCALID", "")
    logging.info(f"HIP_VISIBLE_DEVICES={hip_visible}, ROCR_VISIBLE_DEVICES={rocr_visible}, local_rank={local_rank}, SLURM_LOCALID={slurm_localid}")
    
    # Use SLURM_LOCALID as device ID if available, otherwise use local_rank
    if os.environ.get("SLURM_LOCALID") is not None:
        device_id = int(os.environ.get("SLURM_LOCALID"))
        logging.info(f"Using SLURM_LOCALID={device_id} as device ID")
        
        # Ensure environment variables are set correctly
        if hip_visible != str(device_id):
            os.environ["HIP_VISIBLE_DEVICES"] = str(device_id)
            logging.info(f"Setting HIP_VISIBLE_DEVICES={device_id}")
        
        if rocr_visible != str(device_id):
            os.environ["ROCR_VISIBLE_DEVICES"] = str(device_id)
            logging.info(f"Setting ROCR_VISIBLE_DEVICES={device_id}")
    else:
        device_id = local_rank
        logging.info(f"Using local_rank={device_id} as device ID")
    
    # Check if CUDA is available after setting environment variables
    if not torch.cuda.is_available():
        if os.environ.get("MACHINE") == "frontier":
            logging.warning("CUDA not available on Frontier, but continuing with ROCm backend")
            # On Frontier, we'll try to proceed with ROCm even if CUDA isn't detected
            device = torch.device(f"cuda:{device_id}")
            handle = device_id
            return device, handle
        else:
            raise RuntimeError("No GPU support available. This script requires either NVIDIA CUDA or AMD ROCm GPUs.")
    
    logging.info(f"Attaching GPU {device_id}")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # Log device properties
    try:
        props = torch.cuda.get_device_properties(device_id)
        logging.info(f"Using device: {props.name}, Total memory: {props.total_memory/(1024**3):.2f} GB")
    except Exception as e:
        logging.warning(f"Could not get device properties: {e}")
    
    logging.info(f"Initialized GPU {device_id} on device {device}")
    
    if NVIDIA_AVAILABLE:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        except Exception as e:
            logging.warning(f"Could not initialize NVML: {e}")
            handle = None
    elif ROCM_AVAILABLE:
        handle = device_id  # ROCm SMI uses device index directly
    else:
        handle = None

    return device, handle

def get_profiler():
    """Returns appropriate profiling tools based on GPU backend"""
    if NVIDIA_AVAILABLE:
        return torch.cuda.nvtx