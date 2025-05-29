#!/bin/bash

# Total number of files
TOTAL_FILES=100 # Change this to your total number of files
FILES_PER_SCREEN=20  # Number of files each screen will process

# Function to run batch in screen session
run_screen_batch() {
    START_INDEX=$1
    END_INDEX=$2
    SCREEN_NAME="screen_${START_INDEX}_${END_INDEX}"

    echo "Starting $SCREEN_NAME"

    # Start screen session
    screen -dmS $SCREEN_NAME bash -c "
    for i in \$(seq $START_INDEX $END_INDEX); do
        echo 'Processing index:' \$i
        python MT_1_pion.py \$i
    done
    echo 'Screen $SCREEN_NAME finished'
    exec bash"
}

# Loop through all files and create screens
for ((i=1; i<=TOTAL_FILES; i+=FILES_PER_SCREEN)); do
    END_INDEX=$((i + FILES_PER_SCREEN - 1))
    if [ $END_INDEX -gt $TOTAL_FILES ]; then
        END_INDEX=$TOTAL_FILES
    fi
    run_screen_batch $i $END_INDEX
done

echo "All screens started. Use 'screen -ls' to check running sessions."
