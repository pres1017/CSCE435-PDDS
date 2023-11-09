# CSCE435-PDDS

## 1. Group members:
1. Daniel Armenta
2. Prestone Malaer
3. Shurui Xu
4. Rahul Kumar

---

## 2. _due 10/25_ Sorting

## 2. _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
We will implement sorting algorithms and measure their performance on MPI and CUDA.
- Bitonic(CUDA)
- Bitonic(MPI)
- Merge sort (CUDA)
- Merge sort (MPI)
- Enumeration Sort (CUDA)
- Enumeration Sort (MPI)
- Bubble Sort (CUDA)
- Bubble Sort (MPI)

Enumeration Sort (MPI)
```
MPI_Bcast(unsorted_array) //broadcast unsorted array to all threads
Local_sorted_indices[]  //empty
Sorted_array[] //empty
Unsorted_array[]  //filled

X = size/numprocs
For each section the thread with rank i handles (rank*x<i<(rank+1)*x):
	Count = 0 //sorted index of each element in the array
	For element j in the section, loop through the entire unsorted array:
		If array[i]>array[j] or (array[i]==array[j] and index i > index j):
			Count += 1
	local _unsorted_indices[i-rank*x] = count
MPI_Gather(local_sorted_indices, sorted_indices) 
In the master process (numproc = 0):
	Unsorted_arr_cpy[] = unsorted_array
	For each element i in the sorted array:
		Sorted_array[sorted_indices[i]] = Unsorted_arr_cpy[I]
```
Enumeration Sort (CUDA) 
```
malloc(host memory)
cudaMalloc(device memory)
cudaMemcpy(host_to_device unsorted array)
kernel enumeration_sort<<<BLOCKS, THREADS>>>(device_input, device_output, NUM_VALS);
cudaMemcpy(device_to_host sorted array)
	
enumeration_sort():
	if 'index' is within bounds of the array then
		Initialize 'rank' to zero.
		  For each element 'i' in the array:
		    If the current element is greater than element in the array or in a special case equal but greater in index:
		      Increase 'rank' by one.
		    End if
		  End for
		Place current element in 'output' at position 'rank'.
	END IF
```

