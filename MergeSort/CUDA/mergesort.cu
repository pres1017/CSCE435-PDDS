#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* data_validation = "data_validation";

// Function to merge two subarrays
void merge(float arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    float L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1+ j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

// Function to perform merge sort
void mergeSort(float arr[], int l, int r) {
    if (l < r) {
        int m = l+(r-l)/2;

        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);

        merge(arr, l, m, r);
    }
}

__global__ void gpu_bottom_up_merge(float *source, float *dest, int array_size, int width, int slice)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= slice) return; // out of range

    int start = width*idx*2;
    int middle = min(start + width, array_size);
    int end = min(start + 2*width, array_size);

    int i=start;
    int j=middle;
    int k;

    // merge
    for(k=start; k<end; k++){
        if(i<middle && (j>=end || source[i]<source[j])){
            dest[k] = source[i];
            i++;
        }else{
            dest[k] = source[j];
            j++;
        }
    }
}

void gpu_mergesort(float *source, float *dest, int array_size, int num_processes)
{
    float *d_source, *d_dest;
    int size = array_size * sizeof(float);

    // Allocate space on the device
    cudaMalloc((void**) &d_source, size);
    cudaMalloc((void**) &d_dest, size);

    // Copy the host input data to the device
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(comm_small);
    cudaMemcpy(d_source, source, size, cudaMemcpyHostToDevice);
    CALI_MARK_BEGIN(comm_small);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_END(comm);

    // Loop over width and slices
    int nThreads = num_processes;
    CALI_MARK_BEGIN(comp_large);
    for(int width=1; width<array_size; width*=2){
        int slices = array_size/(2*width);
        int nBlocks = slices/nThreads + ((slices%nThreads)?1:0);
        CALI_MARK_BEGIN(comp_small);
        gpu_bottom_up_merge<<<nBlocks, nThreads>>>(d_source, d_dest, array_size, width, slices);
        CALI_MARK_END(comp_small);
        // Swap the buffers
        float *temp = d_source;
        d_source = d_dest;
        d_dest = temp;
    }
    CALI_MARK_END(comp_large);
    // Copy the sorted array back to the host
    cudaMemcpy(dest, d_source, size, cudaMemcpyDeviceToHost);

    // Free the device memory
    cudaFree(d_source);
    cudaFree(d_dest);
}

int main(int argc, char** argv)
{
    CALI_CXX_MARK_FUNCTION;
    cali::ConfigManager mgr;
    const char* main = "main";
    CALI_MARK_BEGIN(main);
    mgr.start();
    // Initialize random number generator
    srand(time(NULL));

    // Define the size of the array
    std::string numVals = argv[1];
    int array_size = std::stoi(numVals);
    
    std::string type = argv[2];
    const int input = std::stoi(type);
    std::string inputType;
        
    std::string processes = argv[3];
    const int numProcesses = std::stoi(processes);

    // Allocate space for the array on the host
    float *h_array = (float*) malloc(array_size * sizeof(float));

    // Fill the array with random numbers
    CALI_MARK_BEGIN(data_init);
    if(input == 0 || 3){
      float val = 1.0;
      for(int i = 0; i < array_size; i++){
        h_array[i] = val;
        val = val + 1.0;
      }
      inputType = "Sorted";
    }
    
    if(input == 1){
      for(int i=0; i<array_size; i++){
          h_array[i] = ((float)rand()/(float)RAND_MAX) * 1000000.0;
      }
      inputType = "Random";
    }
    
    if(input == 2){
      float val = (float)array_size;
      for(int i = 0; i < array_size; i++){
        h_array[i] = val;
        val = val - 1.0;
      }
      inputType = "ReverseSorted";
    }
    
    if(input == 3){
      for(int i = 0; i < array_size; i++){
        int chance = rand()%100;
        int randIndex = rand()%array_size;
        if(chance <= 1){
          float tempVal = h_array[i];
          h_array[i] = h_array[randIndex];
          h_array[randIndex] = tempVal;
        }
      }
      inputType = "1%perturbed";
    }
    CALI_MARK_END(data_init);

    // Allocate space for the sorted array on the host
    float *h_sorted = (float*) malloc(array_size * sizeof(float));

    // Call the GPU merge sort function
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    gpu_mergesort(h_array, h_sorted, array_size, numProcesses);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Print the sorted array
    CALI_MARK_BEGIN(correctness_check);
    mergeSort(h_sorted, 0, array_size - 1); // sequentially merging the collected subarrays 
    bool isSorted = true;
    for(int i=0; i<array_size - 1; i++){
        if(h_array[i + 1] < h_array[i]){
          isSorted = false;        
        }
    }
    if(isSorted){
      printf("Sorted");
    }else{
      printf("Not sorted");
    }
    printf("\n");
    CALI_MARK_END(correctness_check);
    CALI_MARK_END(main);
    int slices = array_size/(2);
    int nBlocks = slices/numProcesses + ((slices%numProcesses)?1:0);
    // Free the host memory
    free(h_array);
    free(h_sorted);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", array_size); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", numProcesses); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", nBlocks); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    mgr.stop();
    mgr.flush();
}