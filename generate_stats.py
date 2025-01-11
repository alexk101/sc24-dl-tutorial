import h5py
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def calculate_stats_in_chunks(file_paths, chunk_size=10):
    """Calculate running mean and std across multiple large HDF5 files"""
    # Initialize variables for Welford's online algorithm
    count = 0
    mean = None
    M2 = None  # For calculating variance

    # Process each file
    for file_path in tqdm(file_paths, desc="Processing files"):
        with h5py.File(file_path, 'r') as f:
            # Get dataset shape
            data = f['fields']  # Adjust key if different
            total_samples = data.shape[0]
            channels = data.shape[1]
            
            # Initialize arrays if first run
            if mean is None:
                mean = np.zeros((1, channels, 1, 1), dtype=np.float64)
                M2 = np.zeros((1, channels, 1, 1), dtype=np.float64)

            # Process file in chunks
            for i in tqdm(range(0, total_samples, chunk_size), 
                         desc=f"Processing {file_path.name}"):
                end_idx = min(i + chunk_size, total_samples)
                chunk = data[i:end_idx]
                
                # Calculate spatial mean for this chunk (across height and width)
                chunk_spatial_mean = chunk.mean(axis=(0, 2, 3), keepdims=True)
                chunk_count = (end_idx - i)
                
                # Update running statistics using Welford's online algorithm
                delta = chunk_spatial_mean - mean
                mean += delta * chunk_count / (count + chunk_count)
                delta2 = chunk_spatial_mean - mean
                M2 += delta * delta2 * chunk_count
                
                count += chunk_count

    # Calculate final standard deviation
    variance = M2 / count
    std = np.sqrt(variance)
    
    return mean, std

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate global means and standard deviations for HDF5 files')
    parser.add_argument('--data_dir', type=Path, required=True,
                      help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=Path, required=True,
                      help='Directory to save output files')
    parser.add_argument('--chunk_size', type=int, default=10,
                      help='Number of samples to process at once (default: 10)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all HDF5 files
    file_paths = sorted(list(args.data_dir.glob("*.h5")))
    if not file_paths:
        raise ValueError(f"No HDF5 files found in {args.data_dir}")
    
    print(f"Found {len(file_paths)} files to process")
    
    # Calculate statistics
    try:
        means, stds = calculate_stats_in_chunks(file_paths, args.chunk_size)
        
        # Save results
        np.save(args.output_dir / "global_means.npy", means)
        np.save(args.output_dir / "global_stds.npy", stds)
        
        print("\nStatistics saved successfully!")
        print(f"Means shape: {means.shape}")
        print(f"Stds shape: {stds.shape}")
        
        # Print some basic statistics for verification
        print("\nMeans per channel:")
        for i in range(means.shape[1]):
            print(f"Channel {i}: {means[0,i,0,0]:.6f}")
        
        print("\nStds per channel:")
        for i in range(stds.shape[1]):
            print(f"Channel {i}: {stds[0,i,0,0]:.6f}")
            
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    main() 