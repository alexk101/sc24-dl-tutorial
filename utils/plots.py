import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import os
import h5py

def generate_images(fields):
    inp, tar, gen = [x.detach().float().cpu().numpy() for x in fields]
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    plt.title('2m temperature')
    ax[0].imshow(tar[0,2,:,:], cmap="turbo")
    ax[0].set_title("ERA5 target")
    ax[1].imshow(gen[0,2,:,:], cmap="turbo")
    ax[1].set_title("ViT prediction")
    fig.tight_layout()
    return fig

def calculate_layerwise_stdv(predictions, tag="layer_stdv"):
    """
    Calculate the standard deviation for each layer in the predictions.
    
    Args:
        predictions: Tensor of shape [batch_size, channels, height, width]
        tag: Name tag for the logged data
        
    Returns:
        stdv: Tensor of shape [channels, height, width] containing standard deviation values
    """
    # Ensure predictions is a tensor
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions)
    
    # Calculate standard deviation across the batch dimension
    # Shape: [channels, height, width]
    stdv = torch.std(predictions, dim=0)
    
    logging.info(f"Calculated layer-wise standard deviation with shape: {stdv.shape}")
    
    return stdv

def save_stdv_values(stdv, save_dir, step, filename_prefix="stdv_values"):
    """
    Save the standard deviation values to disk for later post-processing.
    
    Args:
        stdv: Tensor of shape [channels, height, width] containing standard deviation values
        save_dir: Directory to save the values
        step: Current step/epoch number
        filename_prefix: Prefix for the saved file
        
    Returns:
        filepath: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if it's a tensor
    if isinstance(stdv, torch.Tensor):
        stdv_np = stdv.detach().cpu().numpy()
    else:
        stdv_np = stdv
    
    # Define file paths for different formats
    h5_path = os.path.join(save_dir, f"{filename_prefix}_step{step}.h5")
    npy_path = os.path.join(save_dir, f"{filename_prefix}_step{step}.npy")
    
    # Save in HDF5 format (good for large arrays and metadata)
    with h5py.File(h5_path, 'w') as f:
        # Create dataset
        dataset = f.create_dataset('stdv', data=stdv_np)
        
        # Add metadata
        dataset.attrs['step'] = step
        dataset.attrs['shape'] = stdv_np.shape
        dataset.attrs['channels'] = stdv_np.shape[0]
        dataset.attrs['height'] = stdv_np.shape[1]
        dataset.attrs['width'] = stdv_np.shape[2]
        dataset.attrs['min'] = float(np.min(stdv_np))
        dataset.attrs['max'] = float(np.max(stdv_np))
        dataset.attrs['mean'] = float(np.mean(stdv_np))
    
    
    logging.info(f"Saved standard deviation values to {h5_path} and {npy_path}")
    
    return h5_path

def log_stdv_image(stdv, writer, global_step, tag="stdv_image", save_dir=None):
    """
    Calculate the average standard deviation across channels and log as an image.
    Also save the values if save_dir is provided.
    
    Args:
        stdv: Tensor of shape [channels, height, width] containing standard deviation values
        writer: TensorBoard SummaryWriter instance
        global_step: Current training step
        tag: Name tag for the logged image
        save_dir: Directory to save the values (if None, values won't be saved)
        
    Returns:
        mean_stdv_np: NumPy array of shape [height, width] containing the mean standard deviation
    """
    # Ensure stdv is a tensor
    if not isinstance(stdv, torch.Tensor):
        stdv = torch.tensor(stdv)
    
    # Calculate mean across channels
    # Shape: [height, width]
    mean_stdv = torch.mean(stdv, dim=0)
    
    # Convert to numpy for matplotlib
    mean_stdv_np = mean_stdv.detach().cpu().numpy()
    
    # Create figure and plot
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    im = ax.imshow(mean_stdv_np, cmap='viridis')
    plt.colorbar(im, ax=ax)
    plt.title(f"Average Standard Deviation Across Channels")
    
    # Log to TensorBoard
    writer.add_figure(tag, fig, global_step)
    
    # Also log as image for simpler viewing
    # Normalize to [0, 1] for image logging
    min_val = mean_stdv_np.min()
    max_val = mean_stdv_np.max()
    normalized = (mean_stdv_np - min_val) / (max_val - min_val + 1e-8)
    
    # Add channel dimension for TensorBoard
    normalized = normalized[None, :, :]  # Shape: [1, height, width]
    
    writer.add_image(f"{tag}_normalized", normalized, global_step)
    
    # Save values if directory is provided
    if save_dir:
        # Save the full 3D stdv array
        save_stdv_values(stdv, save_dir, global_step, f"{tag}_full")
        
        # Also save the mean stdv separately
        mean_save_dir = os.path.join(save_dir, "mean_stdv")
        os.makedirs(mean_save_dir, exist_ok=True)
        np.save(os.path.join(mean_save_dir, f"{tag}_mean_step{global_step}.npy"), mean_stdv_np)
    
    logging.info(f"Logged standard deviation image with shape: {mean_stdv.shape}")
    
    plt.close(fig)
    
    return mean_stdv_np

def load_stdv_values(filepath):
    """
    Load saved standard deviation values.
    
    Args:
        filepath: Path to the saved file (.h5 or .npy)
        
    Returns:
        stdv_values: NumPy array containing the standard deviation values
        metadata: Dictionary of metadata (for HDF5 files only)
    """
    with h5py.File(filepath, 'r') as f:
        # Load data
        stdv_values = f['stdv'][:]
        
        # Load metadata
        metadata = {}
        for key, value in f['stdv'].attrs.items():
            metadata[key] = value
            
        return stdv_values, metadata