import h5py
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging

def setup_logging(output_dir):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'stats_generation.log'),
            logging.StreamHandler()
        ]
    )

def process_chunk(chunk, device):
    """Process a single chunk of data on GPU"""
    # Convert to torch tensor and move to GPU
    chunk_tensor = torch.from_numpy(chunk).to(device)
    
    # Calculate mean across samples, height, and width
    spatial_mean = chunk_tensor.mean(dim=(0, 2, 3), keepdim=True)
    
    # Move result back to CPU and convert to numpy
    return spatial_mean.cpu().numpy(), chunk.shape[0]

def calculate_stats_in_chunks(file_paths, chunk_size=10, num_workers=4, device='cuda'):
    """Calculate running mean and std across multiple large HDF5 files using GPU"""
    count = 0
    mean = None
    M2 = None

    # Use thread pool for I/O operations
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for file_path in tqdm(file_paths, desc="Processing files"):
            with h5py.File(file_path, 'r') as f:
                data = f['fields']
                total_samples = data.shape[0]
                channels = data.shape[1]
                
                # Initialize arrays if first run
                if mean is None:
                    mean = np.zeros((1, channels, 1, 1), dtype=np.float64)
                    M2 = np.zeros((1, channels, 1, 1), dtype=np.float64)

                # Process chunks in parallel
                chunk_futures = []
                for i in range(0, total_samples, chunk_size):
                    end_idx = min(i + chunk_size, total_samples)
                    # Create memory-mapped array for chunk
                    chunk = data[i:end_idx]
                    future = executor.submit(process_chunk, chunk, device)
                    chunk_futures.append(future)

                # Process results as they complete
                for future in tqdm(chunk_futures, desc=f"Processing {file_path.name}"):
                    chunk_spatial_mean, chunk_count = future.result()
                    
                    # Update running statistics (on CPU)
                    delta = chunk_spatial_mean - mean
                    mean += delta * chunk_count / (count + chunk_count)
                    delta2 = chunk_spatial_mean - mean
                    M2 += delta * delta2 * chunk_count
                    count += chunk_count

                # Clear CUDA cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Calculate final standard deviation
    variance = M2 / count
    std = np.sqrt(variance)
    
    return mean, std

def main():
    parser = argparse.ArgumentParser(description='Calculate global means and standard deviations for HDF5 files')
    parser.add_argument('--data_dir', type=Path, required=True,
                      help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=Path, required=True,
                      help='Directory to save output files')
    parser.add_argument('--chunk_size', type=int, default=10,
                      help='Number of samples to process at once (default: 10)')
    parser.add_argument('--num_workers', type=int, 
                      default=min(4, mp.cpu_count()),
                      help='Number of worker threads (default: min(4, cpu_count))')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for computations (cuda/cpu)')
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Create output directory and setup logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    
    # Log GPU info if using CUDA
    if args.device == 'cuda':
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get all HDF5 files
    file_paths = sorted(list(args.data_dir.glob("*.h5")))
    if not file_paths:
        raise ValueError(f"No HDF5 files found in {args.data_dir}")
    
    logging.info(f"Found {len(file_paths)} files to process")
    logging.info(f"Using chunk size: {args.chunk_size}")
    logging.info(f"Using {args.num_workers} workers")
    logging.info(f"Using device: {args.device}")
    
    try:
        # Calculate statistics
        means, stds = calculate_stats_in_chunks(
            file_paths, 
            args.chunk_size,
            args.num_workers,
            args.device
        )
        
        # Save results (no compression needed for small arrays)
        np.save(args.output_dir / "global_means.npy", means)
        np.save(args.output_dir / "global_stds.npy", stds)
        
        logging.info("\nStatistics saved successfully!")
        logging.info(f"Means shape: {means.shape}")
        logging.info(f"Stds shape: {stds.shape}")
        
        # Print statistics
        logging.info("\nMeans per channel:")
        for i in range(means.shape[1]):
            logging.info(f"Channel {i}: {means[0,i,0,0]:.6f}")
        
        logging.info("\nStds per channel:")
        for i in range(stds.shape[1]):
            logging.info(f"Channel {i}: {stds[0,i,0,0]:.6f}")
            
    except Exception as e:
        logging.error(f"Error during processing: {e}", exc_info=True)
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 