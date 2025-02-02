#!/bin/bash

# Array of values for the first parameter 2 4 8 16 32 64 128 256 512 1024
second_params=(64 128 256 512 1024)   
#first_params=(32) 

# Array of values for the second parameter (2^16, 2^18, ..., 2^28)
#first_params
first_params=(65536 262144 1048576 4194304)
# 16777216 67108864)

# Loop through each value of the first parameter
for first in "${first_params[@]}"; do
    # Loop through each value of the second parameter
    for second in "${second_params[@]}"; do
        # Loop through values 0 to 3 for the third parameter
        for third in {0..3}; do

            sbatch cuda.grace_job "$first" "$second" "$third"
        done
    done
done