#!/bin/bash

vals=(262144 1048576 4194304 16777216 67108864 268435456)
inputs=(0 1 2 3)
processes=(1 64 128 256 512 1024)

for var1 in "${vals[@]}"; do
	for var2 in "${inputs[@]}"; do
		for var3 in "${processes[@]}"; do
			sbatch bitonic.grace_job "$var1" "$var2" "$var3"
			echo "Ran job for vals=$var1, inputs=$var2, processes=$var3"
		done
	done
	sleep 30
done
