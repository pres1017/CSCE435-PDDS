#!/bin/bash



# Define the file name you want to check for
file_name=""


# Array of values for the first parameter 2 4 8 16 32 64 128 256 512 1024
#first_params=(64 128)
second_params=(2 4 8 16 32 64 128 256 512 1024)
#first_params=(32) 

# Array of values for the second parameter (2^16, 2^18, ..., 2^28)
first_params=(65536 262144 1048576 4194304)

# Loop through each value of the first parameter
for first in "${first_params[@]}"; do
    # Loop through each value of the second parameter
    for second in "${second_params[@]}"; do
        # Loop through values 0 to 3 for the third parameter
        for third in {0..3}; do
	    file_name="s${third}-p${second}-a${first}.cali"
	    if [ -f "$file_name" ]; then
    		echo "File '$file_name' exists in the directory."
	    else
    		#echo "fd"
		sbatch mpi.grace_job "$first" "$second" "$third"
	    fi
        done
    done
done
