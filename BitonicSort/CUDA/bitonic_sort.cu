/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;
int kernel_calls = 0;
const size_t COMM_SMALL_THRESHOLD = 1 * 1024 * 1024;
const int COMP_SMALL_THRESHOLD = 1024;
int sorting;


const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

float bitonic_sort_step_time_calculated = 0;
float cudaMemcpy_host_to_device_time_calculated = 0;
float cudaMemcpy_device_to_host_time_calculated = 0;


void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

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

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

void data_init(float *arr, int length, int sorting)
{
  CALI_MARK_BEGIN("data_init");
  if (sorting == 0){ 
    srand(time(NULL));
    for (int j = 0; j < length; j++) {
      arr[j] = random_float();
    }
  } else if (sorting == 1){ 
    for (int j = 0; j < length; j++) {
        arr[j] = j * 1.0;
    }
  } else if (sorting == 2){ 
    for (int j = 0; j < length; j++) {
        arr[j] = length - j * 1.0;
    }
  } else if (sorting == 3){
    for (int j = 0; j < length; j++) {
        arr[j] = j * 1.0;
    }
    srand((unsigned)time(NULL));

    int pCount = length / 100;

    for (int j = 0; j < pCount; j++) {
        int index1 = rand() % length;
        int index2 = rand() % length;

        float t = arr[index1];
        arr[index1] = arr[index2];
        arr[index2] = t;
    }
  }
  CALI_MARK_END("data_init");
}

void correctness_check(float *arr, int l) {
  CALI_MARK_BEGIN("correctness_check");

  bool sorted = true;
  for (int i = 0; i < l - 1; ++i) {
    if (arr[i] > arr[i + 1]) {
      sorted = false;
      break;
    }
  }

  if (!sorted) {
    printf("The array is not sorted");
  } else {
    printf("The array is correctly sorted");
  }

  CALI_MARK_END("correctness_check");
}



__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;
  

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);
  cudaEvent_t start, stop;


  cudaMalloc((void**) &dev_values, size);
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  CALI_MARK_BEGIN("comm");

  if (size < COMM_SMALL_THRESHOLD) {
  CALI_MARK_BEGIN("comm_small");
  } else {
  CALI_MARK_BEGIN("comm_large");
  }

  CALI_MARK_BEGIN(cudaMemcpy_host_to_device);
  cudaEventRecord(start);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  CALI_MARK_END(cudaMemcpy_host_to_device);

  if (size < COMM_SMALL_THRESHOLD) {
  CALI_MARK_END("comm_small");
  } else {
  CALI_MARK_END("comm_large");
  }

  CALI_MARK_END("comm");

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time_calculated, start, stop);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  /* Major step */

  CALI_MARK_BEGIN("comp");
  if (THREADS < COMP_SMALL_THRESHOLD) {
    CALI_MARK_BEGIN("comp_small");
  } else {
    CALI_MARK_BEGIN("comp_large");
  }

  CALI_MARK_BEGIN(bitonic_sort_step_region);
  cudaEventRecord(start);
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      kernel_calls++;
    
    }
  }
  cudaEventRecord(stop);
  CALI_MARK_END(bitonic_sort_step_region);

  if (THREADS < COMP_SMALL_THRESHOLD) {
    CALI_MARK_END("comp_small");
  } else {
    CALI_MARK_END("comp_large");
  }
  CALI_MARK_END("comp");

  cudaDeviceSynchronize();
  cudaEventElapsedTime(&bitonic_sort_step_time_calculated, start, stop);
  
  
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN("comm");

  if (size < COMM_SMALL_THRESHOLD) {
  CALI_MARK_BEGIN("comm_small");
  } else {
  CALI_MARK_BEGIN("comm_large");
  }

  CALI_MARK_BEGIN(cudaMemcpy_device_to_host);
  cudaEventRecord(start);
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  CALI_MARK_END(cudaMemcpy_device_to_host);

  if (size < COMM_SMALL_THRESHOLD) {
  CALI_MARK_END("comm_small");
  } else {
  CALI_MARK_END("comm_large");
  }

  CALI_MARK_END("comm");

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time_calculated, start, stop);
  
  cudaFree(dev_values);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(int argc, char *argv[])
{
  CALI_CXX_MARK_FUNCTION;
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;
  sorting = atoi(argv[3]);

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  data_init(values, NUM_VALS, sorting); 
  

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
  correctness_check(values, NUM_VALS);



  // Adiak metadata collection
    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "BitonicSort");
    adiak::value("ProgrammingModel", "CUDA");
    adiak::value("Datatype", "float");
    adiak::value("SizeOfDatatype", sizeof(float));
    adiak::value("InputSize", NUM_VALS);

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

    adiak::value("num_procs", THREADS); 
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("group_num", 20); 
    adiak::value("implementation_source", "Online"); 


  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}