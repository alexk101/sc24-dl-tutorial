import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import math

# Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ == '__main__':
    mp.set_start_method('spawn')

def setup_logging(output_dir):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'resize_dataset.log'),
            logging.StreamHandler()
        ]
    )

def process_chunk(chunk_data, device='cuda'):
    """Process a chunk of data using GPU if available"""
    if device == 'cuda' and torch.cuda.is_available():
        # Move to GPU, resize, and back to CPU
        with torch.cuda.amp.autocast():
            chunk_tensor = torch.from_numpy(chunk_data).to(device)
            resized = chunk_tensor[:, :, :720, :]
            return resized.cpu().numpy()
    else:
        # Process on CPU
        return chunk_data[:, :, :720, :]

def resize_file(input_file: Path, output_file: Path, chunk_size: int, device: str):
    """Resize a single HDF5 file by removing the last row"""
    with h5py.File(input_file, 'r') as f_in:
        data = f_in['fields']
        original_shape = data.shape
        
        # Calculate new shape and chunks
        new_shape = list(original_shape)
        new_shape[-2] = 720
        
        # Calculate optimal chunk size for HDF5
        chunk_shape = (
            min(chunk_size, new_shape[0]),  # samples
            new_shape[1],                   # channels
            min(720, 128),                  # height
            min(new_shape[3], 128)          # width
        )
        
        with h5py.File(output_file, 'w') as f_out:
            # Create resized dataset with optimized chunking
            resized = f_out.create_dataset(
                'fields', 
                shape=new_shape,
                dtype=data.dtype,
                chunks=chunk_shape,
                compression='lzf'  # Fast, lossless compression
            )
            
            # Process data in chunks
            for i in range(0, original_shape[0], chunk_size):
                end_idx = min(i + chunk_size, original_shape[0])
                chunk_data = data[i:end_idx]
                resized_chunk = process_chunk(chunk_data, device)
                resized[i:end_idx] = resized_chunk
            
            # Copy attributes
            for key, value in f_in['fields'].attrs.items():
                resized.attrs[key] = value

def process_file_wrapper(args):
    """Wrapper function for parallel processing"""
    input_file, output_file, chunk_size, device = args
    try:
        resize_file(input_file, output_file, chunk_size, device)
        return True, input_file.name
    except Exception as e:
        return False, f"{input_file.name}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Resize ERA5 dataset from 721 to 720 height')
    parser.add_argument('--input_dir', type=Path, required=True,
                      help='Directory containing original HDF5 files')
    parser.add_argument('--output_dir', type=Path, required=True,
                      help='Directory to save resized files')
    parser.add_argument('--chunk_size', type=int, default=32,
                      help='Number of samples to process at once')
    parser.add_argument('--num_workers', type=int, 
                      default=max(1, mp.cpu_count() // 2),
                      help='Number of parallel processes')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for computations (cuda/cpu)')
    args = parser.parse_args()
    
    # Create output directory and setup logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Get all HDF5 files
    input_files = sorted(list(args.input_dir.glob("*.h5")))
    if not input_files:
        raise ValueError(f"No HDF5 files found in {args.input_dir}")
    
    logging.info(f"Found {len(input_files)} files to process")
    logging.info(f"Using device: {args.device}")
    logging.info(f"Using {args.num_workers} workers")
    logging.info(f"Chunk size: {args.chunk_size}")
    
    # Prepare arguments for parallel processing
    process_args = [
        (input_file, args.output_dir / input_file.name, args.chunk_size, args.device)
        for input_file in input_files
    ]
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_file_wrapper, process_args),
            total=len(input_files),
            desc="Processing files"
        ))
    
    # Log results
    successes = sum(1 for success, _ in results if success)
    failures = [(msg, msg) for success, msg in results if not success]
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successfully processed: {successes}/{len(input_files)} files")
    
    if failures:
        logging.error("\nFailed files:")
        for failure_msg, error_msg in failures:
            logging.error(error_msg)

if __name__ == "__main__":
    main() 