# CSCE435-PDDS
## 0.Group Number: 20

## 1. Group members:
1. Daniel Armenta
2. Prestone Malaer
3. Shurui Xu
4. Rahul Kumar

---

## 2. _due 10/25_ Sorting

### 2a _due 10/25_ Brief project description (what algorithms will you be comparing and on what architectures)
We will implement sorting algorithms and measure their performance on MPI and CUDA.
- Bitonic(CUDA)
- Bitonic(MPI)
- Merge sort (CUDA)
- Merge sort (MPI)
- Enumeration Sort (CUDA)
- Enumeration Sort (MPI)
- Bubble Sort (CUDA)
- Bubble Sort (MPI)

### 2b. Pseudocode for each parallel algorithm
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
Bubble/Odd-Even Sort (MPI)
```
bubble(local_arr, size_local_arr) {
    holder_arr = [size_local_arr]
    for(0 to num_processors-1) {
        if (taskid % 2 == 1 and num_processors % 2 == 1) { 
            if( in odd phase) { 
                send local_arr to process taskid + 1 using MPI_Send
                receive local_arr from process taskid + 1 using MPI_Recv and put into holder_arr variable
                this process keeps the lower half of numbers from local_arr + holder_arr
            }
            else if (in even phase and taskid != num_processors - 1) {//in even phase - exclude last
                send local_arr to process taskid - 1 using MPI_Send
                receive local_arr from process taskid - 1 using MPI_Recv and put into holder_arr variable
                this process keeps the higher half of numbers from local_arr + holder_arr

            }
        }
        // first and last processes do not get a pair
        else if(taskid % 2 == 1 && num_processors % 2 == 0) {

            if (in odd phase and rank != numtasks - 1) { 
                send local_arr to process taskid + 1 using MPI_Send
                receive local_arr from process taskid + 1 using MPI_Recv and put into holder_arr variable
                this process keeps the lower half of numbers from local_arr + holder_arr
            }
            else if (in even phase) { 
                send local_arr to process taskid - 1 using MPI_Send
                receive local_arr from process taskid - 1 using MPI_Recv and put into holder_arr variable
                this process keeps the higher half of numbers from local_arr + holder_arr
            }
        }

        else if(taskid % 2 == 0 && num_processors % 2 == 1) {

            if (in odd phase and taskid != 0) { 
                send local_arr to process taskid - 1 using MPI_Send
                receive local_arr from process taskid - 1 using MPI_Recv and put into holder_arr variable
                this process keeps the higher half of numbers from local_arr + holder_arr
            }
            else if (in even phase and taskid != numtasks - 1) { //in even phase - exclude last
                send local_arr to process taskid + 1 using MPI_Send
                receive local_arr from process taskid + 1 using MPI_Recv and put into holder_arr variable
                this process keeps the lower half of numbers from local_arr + holder_arr
            }
        }
        else { // taskid % 2 == 0 and num_processors % 2 == 0  
            if (in odd phase and taskid != 0) { 
                send local_arr to process taskid - 1 using MPI_Send
                receive local_arr from process taskid - 1 using MPI_Recv and put into holder_arr variable
                this process keeps the higher half of numbers from local_arr + holder_arr
            }
            else if (in even phase) { 
                send local_arr to process taskid + 1 using MPI_Send
                receive local_arr from process taskid + 1 using MPI_Recv and put into holder_arr variable
                this process keeps the lower half of numbers from local_arr + holder_arr
            }
        }
    }
    gather all the arrays into master using MPI_Gather
}
```




## 3

MERGESORT MPI
```
main():
    MPI_Init(&argc, &argv);


    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    const int n = std::stoi(arg);
    int array[n];


    MPI_Bcast(array, n, MPI_INT, 0, MPI_COMM_WORLD);


    int local_n = n / size;
    int local_array[local_n];


    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);


    mergeSort(local_array, 0, local_n - 1);


    MPI_Gather(local_array, local_n, MPI_INT, array, local_n, MPI_INT, 0, MPI_COMM_WORLD);


/*
Scatters the arrays to the threads
Performs MergeSort
Gathers the arrays back in main
*/
	
```

MERGESORT CUDA
```
void parallelMergeSort(int *a, int *b, int n) {
    int *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));


    cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
   
    for (int i = 2; i <= n; i *= 2) {
        for (int j = 0; j < n; j += i) {
            CALIPER_MARK_BEGIN(small_comp);
            merge<<<numBlocks, blockSize>>>(dev_a, dev_b, dev_b, i);
            cudaDeviceSynchronize();
            CALIPER_MARK_END(small_comp);
        }
    }


    cudaMemcpy(a, dev_a, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
}


int main() {
    cali::ConfigManager mgr;
    mgr.start();


    int a[] = {12, 11, 13, 5, 6, 7};
    int n = sizeof(a)/sizeof(a[0]);
    int *b = (int*)malloc(n * sizeof(int));


    parallelMergeSort(a, b, n);


    free(b);
```

### 2c. Evaluation plan - what and how will you measure and compare
```
Input sizes: 2^16, 2^20, 2^24
Input types: Float
Strong scaling (same problem size, increase number of processors/nodes)
Weak scaling (increase problem size, increase number of processors)
Number of threads in a block on the GPU
```
## 3. Project implementation

Please see GitHub branches for algorithm implementations and Caliper files. 

