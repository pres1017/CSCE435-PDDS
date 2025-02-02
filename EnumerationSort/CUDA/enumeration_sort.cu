#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;
int sorting;
int operations = 0;

const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* cuda_Memcpy = "cudaMemcpy";


// Store results in these variables.
float effective_bandwidth_gb_s = 0.0;
float bitonic_sort_step_time = 0.0;
float cudaMemcpy_host_to_device_time = 0.0;
float cudaMemcpy_device_to_host_time = 0.0;

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void data_init(float *arr, int length, int sorting)
{
  CALI_CXX_MARK_FUNCTION;
  if (sorting == 0){ //random sort
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
      arr[i] = random_float();
    }
  }

  else if (sorting == 1){ //sorted
    float step = 1.0; 
    for (int i = 0; i < length; i++) {
        arr[i] = i * step;
    }
  } 

  else if (sorting == 2){ //reverse sorted
    float step = 1.0; 
    for (int i = 0; i < length; i++) {
        arr[i] = length - i * step;
    }
  }

  else if (sorting == 3){
    float step = 1.0; // Step size for the sorted array

    // Initialize the array with sorted values
    for (int i = 0; i < length; i++) {
        arr[i] = i * step;
    }

    srand((unsigned)time(NULL));

    // Calculate 1% of the array's length
    int perturbCount = length / 100;

    for (int i = 0; i < perturbCount; i++) {
        // Select two random indices
        int idx1 = rand() % length;
        int idx2 = rand() % length;

        // Swap the elements at these indices
        float temp = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = temp;
    }
  }
}

bool correctness_check(float *arr, int length){
  CALI_CXX_MARK_FUNCTION;
  bool sorted = true;
  for (int i = 0; i < length-1; ++i) {
    if(arr[i]>arr[i+1]){
      sorted = false;
      break;
    }
  }
  return sorted;
}

__global__ void enumeration_sort(float *input, float *output, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    int rank = 0;
    for (int i = 0; i < n; ++i) {
      if (input[index] > input[i] || (input[index] == input[i] && index > i)) {
        rank++;
      }
    }
    output[rank] = input[index];
  }
}


int main(int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    THREADS = atoi(argv[1]);
    NUM_VALS = atoi(argv[2]);
    sorting = atoi(argv[3]);
    BLOCKS = NUM_VALS / THREADS;
    size_t size = NUM_VALS * sizeof(float);

    // printf("Number of threads: %d\n", THREADS);
    // printf("Number of values: %d\n", NUM_VALS);
    // printf("Number of blocks: %d\n", BLOCKS);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    // allocate host memory
    float *host_input = (float*)malloc(size);
    float *host_output = (float*)malloc(size);
    data_init(host_input, NUM_VALS, sorting);

    // allocate device memory
    float *device_input, *device_output;
    cudaMalloc((void**)&device_input, size);
    cudaMalloc((void**)&device_output, size);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cuda_Memcpy);
    // copy from host to device
    cudaMemcpy(device_input, host_input, size, cudaMemcpyHostToDevice);
    CALI_MARK_END(cuda_Memcpy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    // kernel
    enumeration_sort<<<BLOCKS, THREADS>>>(device_input, device_output, NUM_VALS);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(cuda_Memcpy);
    // copy from device to host
    cudaMemcpy(host_output, device_output, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(cuda_Memcpy);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    //printf("Random Array:\n");
    //array_print(host_input, NUM_VALS);
    //printf("Sorted Array:\n");
    //array_print(host_output, NUM_VALS);
    bool sorted = correctness_check(host_output, NUM_VALS);
    printf("\nSorted is: %s\n", sorted ? "true" : "false");
    


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    switch (sorting) {
      case 0:
        adiak::value("InputType", "Random");
        break;
      case 1:
        adiak::value("InputType", "Sorted");
        break;
      case 2:
        adiak::value("InputType", "ReverseSorted");
        break;
      case 3:
        adiak::value("InputType", "1%%perturbed");
        break;
      default:
        adiak::value("InputType", "Random");
     }
    //adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
}