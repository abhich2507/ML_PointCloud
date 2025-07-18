#!/bin/bash

# Define list of x y z combinations
configs=(
  "25 25 200"
  "50 50 25"
  "50 50 50"
  "50 50 100"
  "50 50 200"
)

# Loop through each config
for config in "${configs[@]}"; do
  echo "=============================="
  echo "Starting config: $config"
  echo "Time: $(date)"
  echo "------------------------------"
  
  ./run_pion_custom_start_end.sh 1 100 $config
  wait

  ./run_proton_custom_start_end.sh 1 100 $config
  wait

  echo "Finished config: $config"
  echo "Time: $(date)"
  echo "=============================="
  echo ""

  # Sleep for 2 hours (7200 seconds) before the next config
  echo "Sleeping for 45 minutes before next iteration..."
  sleep 2700
done
