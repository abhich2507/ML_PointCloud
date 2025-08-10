#!/bin/bash

# Define list of x y z combinations
configs=(
  "100 100 25"
  "25 25 25"
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
  sleep 800
done
