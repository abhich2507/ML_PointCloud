#!/usr/bin/env python3
"""
Training Runner Script
Runs train_point_cloud.py with multiple energy levels and segmentations.

Usage:
    python run_training.py
"""

import os
import subprocess
import sys
from itertools import product
import time

def run_command(command, timeout=None):
    """
    Run a command and return success status.
    
    Args:
        command (list): Command to run as a list of strings
        timeout (int): Timeout in seconds (None for no timeout)
    
    Returns:
        bool: True if command succeeded, False otherwise
    """
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(command, timeout=timeout, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Success: {' '.join(command[:3])}")
            return True
        else:
            print(f"✗ Failed: {' '.join(command[:3])}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout: {' '.join(command[:3])}")
        return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False

def check_data_exists(energy, granularity):
    """
    Check if data directories exist for given energy and granularity.
    
    Args:
        energy (str): Energy value
        granularity (str): Granularity value
    
    Returns:
        bool: True if both proton and pion directories exist
    """
    pion_dir = f"/mnt/c/Users/hnayak/Documents/{energy}GeV/Pion/small_{granularity}"
    proton_dir = f"/mnt/c/Users/hnayak/Documents/{energy}GeV/Proton/small_{granularity}"
    
    return os.path.exists(pion_dir) and os.path.exists(proton_dir)

def main():
    """Main function to run training with all configurations."""
    
    # Configuration
    training_configs = {
        "10GeV": {
            "granularities": ["25_25_25", "25_25_50", "25_25_100", "25_25_200", 
                             "50_50_25", "50_50_50", "50_50_100", "50_50_200",
                             "100_100_25", "100_100_200", "200_200_50"],
            "epochs": 60,
            "lr": 5e-4,
            "batch_size": 32,
            "num_workers": 32
        }
        # "25GeV": {
        #     "granularities": ["50", "100", "200"],
        #     "epochs": 60,
        #     "lr": 5e-4,
        #     "batch_size": 32,
        #     "num_workers": 32
        # },
        # "50GeV": {
        #     "granularities": ["50", "100", "200"],
        #     "epochs": 60,
        #     "lr": 5e-4,
        #     "batch_size": 32,
        #     "num_workers": 32
        # },
        # "100GeV": {
        #     "granularities": ["100", "200"],
        #     "epochs": 60,
        #     "lr": 5e-4,
        #     "batch_size": 32,
        #     "num_workers": 32
        # }
    }
    
    # Track results
    successful_runs = []
    failed_runs = []
    skipped_runs = []
    
    # Get total number of configurations
    total_configs = sum(len(config["granularities"]) for config in training_configs.values())
    current_config = 0
    
    print(f"Starting training for {total_configs} configurations...")
    print("="*60)
    
    # Run training for each configuration
    for energy, config in training_configs.items():
        print(f"\nProcessing {energy}")
        print("-" * 40)
        
        energy_num = energy.replace("GeV", "")  # Remove "GeV" suffix
        
        for granularity in config["granularities"]:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] {energy} - {granularity}")
            
            # Check if data exists
            if not check_data_exists(energy_num, granularity):
                print(f"⚠ Skipping: Data not found for {energy} {granularity}")
                skipped_runs.append((energy, granularity, "Data not found"))
                continue
            
            # Build command
            command = [
                "python", "train_point_cloud.py",
                energy_num,  # Energy without "GeV"
                granularity,
                "--epochs", str(config["epochs"]),
                "--lr", str(config["lr"]),
                "--batch_size", str(config["batch_size"]),
                "--num_workers", str(config["num_workers"])
            ]
            
            # Record start time
            start_time = time.time()
            
            # Run training
            success = run_command(command, timeout=800)  # 15 minute timeout
            
            # Record end time
            end_time = time.time()
            duration = end_time - start_time
            
            # Track results
            if success:
                successful_runs.append((energy, granularity, f"{duration:.1f}s"))
                print(f"✓ Completed in {duration:.1f} seconds")
            else:
                failed_runs.append((energy, granularity, f"Failed after {duration:.1f}s"))
                print(f"✗ Failed after {duration:.1f} seconds")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"\n✓ Successful runs: {len(successful_runs)}")
    for energy, granularity, duration in successful_runs:
        print(f"  {energy} {granularity} ({duration})")
    
    print(f"\n✗ Failed runs: {len(failed_runs)}")
    for energy, granularity, info in failed_runs:
        print(f"  {energy} {granularity} ({info})")
    
    print(f"\n⚠ Skipped runs: {len(skipped_runs)}")
    for energy, granularity, reason in skipped_runs:
        print(f"  {energy} {granularity} ({reason})")
    
    print(f"\nTotal: {len(successful_runs) + len(failed_runs) + len(skipped_runs)} configurations")
    
    # Save summary to file
    with open("training_summary.txt", "w") as f:
        f.write("TRAINING SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Successful runs: {len(successful_runs)}\n")
        for energy, granularity, duration in successful_runs:
            f.write(f"  {energy} {granularity} ({duration})\n")
        
        f.write(f"\nFailed runs: {len(failed_runs)}\n")
        for energy, granularity, info in failed_runs:
            f.write(f"  {energy} {granularity} ({info})\n")
        
        f.write(f"\nSkipped runs: {len(skipped_runs)}\n")
        for energy, granularity, reason in skipped_runs:
            f.write(f"  {energy} {granularity} ({reason})\n")
    
    print("\nSummary saved to training_summary.txt")

if __name__ == "__main__":
    # Check if train_point_cloud.py exists
    if not os.path.exists("train_point_cloud.py"):
        print("Error: train_point_cloud.py not found in current directory")
        sys.exit(1)
    
    main()