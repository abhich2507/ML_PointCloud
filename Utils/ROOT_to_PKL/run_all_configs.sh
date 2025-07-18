#!/bin/bash

# List of x, y, z combinations
configs=(
  "25 25 100"
  "25 25 200"
  "50 50 25"
  "50 50 50"
  "50 50 100"
  "50 50 200"
)

# Loop and run the main script
for config in "${configs[@]}"; do
  ./run_pion_custom_start_end.sh 1 100 $config
  ./run_proton_custom_start_end.sh 1 100 $config
done
