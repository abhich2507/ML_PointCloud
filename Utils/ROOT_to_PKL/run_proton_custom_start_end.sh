#!/bin/bash

# Usage: ./script.sh <start_index> <end_index>
# Example: ./script.sh 17 36

# Initialize conda for bash shell
echo "Initializing conda and activating root_env..."
source /mnt/miniconda3/etc/profile.d/conda.sh

# Deactivate any active environment and activate root_env
conda deactivate 2>/dev/null || true
conda activate root_env

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_index> <end_index>"
    exit 1
fi

START_TOTAL=$1
END_TOTAL=$2

FILES_PER_SCREEN=5 # Number of files each screen will process

# Function to run batch in screen session
run_screen_batch() {
    START_INDEX=$1
    END_INDEX=$2
    SCREEN_NAME="screen_${START_INDEX}_${END_INDEX}"

    echo "Starting $SCREEN_NAME"

    # Start screen session
    screen -dmS $SCREEN_NAME bash -c "
    # Initialize conda and activate root_env in the screen session
    source /mnt/miniconda3/etc/profile.d/conda.sh
    conda activate root_env
    
    for i in \$(seq $START_INDEX $END_INDEX); do
        echo 'Processing index:' \$i
        python MT_1_proton_50.py \$i
    done
    echo 'Screen $SCREEN_NAME finished'
    exec bash"
}

# Loop through specified range and create screens
for ((i=START_TOTAL; i<=END_TOTAL; i+=FILES_PER_SCREEN)); do
    BATCH_END=$((i + FILES_PER_SCREEN - 1))
    if [ $BATCH_END -gt $END_TOTAL ]; then
        BATCH_END=$END_TOTAL
    fi
    run_screen_batch $i $BATCH_END
done

echo "All screens from $START_TOTAL to $END_TOTAL started."
echo "Use 'screen -ls' to check running sessions."