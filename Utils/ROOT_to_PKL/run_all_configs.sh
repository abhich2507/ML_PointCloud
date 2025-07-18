#!/bin/bash

# Define list of x y z combinations
configs=(
  "25 25 100"
  "25 25 200"
  "50 50 25"
  "50 50 50"
  "50 50 100"
  "50 50 200"
)

# Loop through each config
for config in "${configs[@]}"; do
  echo "=============================="
  echo "Running for config: $config"
  echo "------------------------------"
  
  ./run_pion_custom_start_end.sh 1 100 $config
  wait  # Waits for the above to finish (redundant here, but safe)

  ./run_proton_custom_start_end.sh 1 100 $config
  wait

  echo "Finished config: $config"
  echo "=============================="
  echo ""
done
