#!/usr/bin/env python3
"""
Pickle File Splitter Script
Splits large pickle files containing multiple DataFrames into individual pickle files.
Supports multiple segmentations and energy levels.

Usage:
    python Splitter_PKL.py
"""

import os
import pickle
import glob
import tqdm
import multiprocessing as mp
from functools import partial

def save_dataframe(df_with_index, output_dir, particle_prefix):
    """Save a single DataFrame to a pickle file."""
    df, index = df_with_index
    filename = os.path.join(output_dir, f"s{particle_prefix.lower()}{index}.pkl")
    df.to_pickle(filename)

def process_file(index_offset, pkl_file):
    """Process a single pickle file and return DataFrames with their indices."""
    with open(pkl_file, "rb") as f:
        dataframes = pickle.load(f)
    
    # Pair each df with its global index
    return [(df, index_offset + i) for i, df in enumerate(dataframes, start=1)]

def split_pickle_files(input_dir, output_dir, particle_name, use_multiprocessing=True):
    """
    Split pickle files from input directory into individual files in output directory.
    
    Args:
        input_dir (str): Path to input directory containing large pickle files
        output_dir (str): Path to output directory for individual pickle files
        particle_name (str): Name of particle (e.g., 'proton', 'pion')
        use_multiprocessing (bool): Whether to use multiprocessing for faster execution
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pickle files in sorted order
    pkl_files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    
    if not pkl_files:
        print(f"No pickle files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} pickle files to process")
    
    if use_multiprocessing:
        # Parallel processing
        all_pairs = []
        print("Loading data from .pkl files...")
        
        index_offset = 1
        for pkl_file in tqdm.tqdm(pkl_files, desc="Reading files"):
            df_with_indices = process_file(index_offset, pkl_file)
            all_pairs.extend(df_with_indices)
            index_offset += len(df_with_indices)
        
        print(f"Saving {len(all_pairs)} DataFrames using multiprocessing...")
        
        with mp.Pool(mp.cpu_count()) as pool:
            pool.map(partial(save_dataframe, output_dir=output_dir, particle_prefix=particle_name), all_pairs)
    
    else:
        # Sequential processing
        global_counter = 1
        
        for idx, pkl_file in tqdm.tqdm(enumerate(pkl_files, start=1), total=len(pkl_files)):
            with open(pkl_file, "rb") as f:
                dataframes = pickle.load(f)
            
            # Save each DataFrame to a pickle file
            for df in dataframes:
                filename = os.path.join(output_dir, f"s{particle_name.lower()}{global_counter}.pkl")
                df.to_pickle(filename)
                global_counter += 1
    
    print("Splitting and saving completed successfully!")

def main():
    """Main function to process multiple segmentations and energy levels."""
    
    # Configuration
    base_input_dir = "/mnt/c/Users/hnayak/Documents"
    base_output_dir = "/mnt/c/Users/hnayak/Documents"
    
    # Define energy levels and segmentations to process
    config = {
        "10GeV": {
            "segmentations": ["25_25_25", "25_25_50","25_25_100", "25_25_200", "50_50_25", "50_50_50","50_50_100", "50_50_200"
                              ,"200_200_50","100_100_25", "100_100_200"],
            "particles": ["Proton", "Pion"]
        }
    }
    
    # Process each energy level
    for energy, energy_config in config.items():
        print(f"\n{'='*50}")
        print(f"Processing {energy}")
        print(f"{'='*50}")
        
        for particle in energy_config["particles"]:
            for segmentation in energy_config["segmentations"]:
                print(f"\nProcessing {particle} with segmentation {segmentation}")
                
                # Construct input and output directories
                if energy == "10GeV":
                    input_dir = os.path.join(base_input_dir, energy, particle, segmentation)
                    output_dir = os.path.join(base_output_dir, energy, particle, f"small_{segmentation}")
                else:
                    input_dir = os.path.join(base_input_dir, f"PKL_{particle}_{energy}_{segmentation}")
                    output_dir = os.path.join(base_output_dir, f"small_PKL_{particle}_{energy}_{segmentation}")
                
                # Check if input directory exists
                if not os.path.exists(input_dir):
                    print(f"Warning: Input directory does not exist: {input_dir}")
                    continue
                
                print(f"Input:  {input_dir}")
                print(f"Output: {output_dir}")
                
                # Process the files
                try:
                    split_pickle_files(input_dir, output_dir, particle, use_multiprocessing=True)
                except Exception as e:
                    print(f"Error processing {particle} {segmentation}: {e}")
                    continue

if __name__ == "__main__":
    main()