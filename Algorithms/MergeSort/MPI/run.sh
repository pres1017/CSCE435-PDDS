#!/bin/bash

vals=(16777216)
inputs=(0 1 2 3)
processes=(1024)

for var1 in "${vals[@]}"; do
	for var2 in "${inputs[@]}"; do
		for var3 in "${processes[@]}"; do
			sbatch mpi.grace_job "$var1" "$var2" "$var3"
			echo "Ran for vals=$var1, inputs=$var2, processes=$var3"
		done
	done
	sleep 45
done
