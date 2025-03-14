#!/usr/bin/env python3

import os
import re
import glob
import argparse

def parse_logs(log_dir, search_string, output_file):
    """
    Parse log files in the given directory, search for a specific string,
    and output results to a file.
    
    Args:
        log_dir: Directory containing log files
        search_string: String to search for in each log file
        output_file: File to write results to
    """
    # Get all log files matching the pattern
    log_pattern = os.path.join(log_dir, "dist_test_rank*.log")
    log_files = glob.glob(log_pattern)
    
    # Extract rank numbers from filenames
    rank_pattern = re.compile(r'dist_test_rank(\d+)\.log')
    results = []
    
    for log_file in log_files:
        # Extract rank number from filename
        match = rank_pattern.search(log_file)
        if match:
            rank = int(match.group(1))
            
            # Search for the string in the file
            found = False
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    if search_string in content:
                        found = True
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
            
            results.append((rank, found))
    
    # Sort results by rank
    results.sort(key=lambda x: x[0])
    
    # Write results to output file
    with open(output_file, 'w') as f:
        f.write(f"Search results for string: '{search_string}'\n")
        f.write("-" * 50 + "\n")
        for rank, found in results:
            status = "FOUND" if found else "NOT FOUND"
            f.write(f"Rank {rank}: {status}\n")
        
        # Summary
        total_ranks = len(results)
        found_count = sum(1 for _, found in results if found)
        f.write("-" * 50 + "\n")
        f.write(f"Summary: String found in {found_count} out of {total_ranks} rank logs\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse distributed training log files")
    parser.add_argument("--log_dir", default=".", help="Directory containing log files")
    parser.add_argument("--search_string", default="broadcast complete 200/1", 
                        help="String to search for in log files")
    parser.add_argument("--output_file", default="search_results.txt", 
                        help="File to write results to")
    
    args = parser.parse_args()
    
    parse_logs(args.log_dir, args.search_string, args.output_file)
    print(f"Results written to {args.output_file}")
