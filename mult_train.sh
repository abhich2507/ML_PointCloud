#!/bin/bash
# filepath: /mnt/c/Users/hnayak/Documents/ML_PointCloud/mult_train.sh

# Simple training script with timeout
granularities=( "50_50_50" "50_50_100" "50_50_200" "100_100_25" "100_100_200" "200_200_50")

for granularity in "${granularities[@]}"; do
    echo "Training with granularity: $granularity"
    
    python train_point_cloud.py 10 "$granularity"
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $granularity"
    elif [ $? -eq 124 ]; then
        echo "✗ Timeout: $granularity"
    else
        echo "✗ Failed: $granularity"
    fi
    echo "---"
    # echo "Pausing for 1 hour 15 minutes..."
    # sleep 1h15m
done

echo "All training completed!"