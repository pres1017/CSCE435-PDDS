//split the array by len(array) / numthreads. <-- this the size that each of the array that each thread is getting
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

#define MAXBOUND 1000001  // up to a million


using std::cout, std::endl, std::string;

const char* main = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";


void Correctness(double* arr, int size) {
    for(int i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            cout << "Not sorted" << endl;
            return;
        }
    }
    cout << "Sorted" << endl;
}

__global__ void oddEvenSort_step(double* data, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x; // Total number of threads
    double swap;

    for (int i = 0; i < size; i++) {
        for (int j = tid; j < size - 1; j += step) {
            if (i % 2 == 0) { // Even phase
                if (j % 2 == 0 && data[j] > data[j + 1]) {
                    // Swap adjacent elements
                    swap = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = swap;
                }
            } else { // Odd phase
                if (j % 2 != 0 && data[j] > data[j + 1]) {
                    // Swap adjacent elements
                    swap = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = swap;
                }
            }
        }

        __syncthreads(); // Synchronize threads within the block
    }
}

void SetSeed() {
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    std::srand(seed);
}
void oddEvenSort(double* arr) {
    float *dev_values;
    size_t size = NUM_VALS * sizeof(double);

    cudaMalloc((void**) &dev_values, size);
    
    //MEM COPY FROM HOST TO DEVICE
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_values, arr, size, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(THREADS,1);  /* Number of threads  */
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    oddEvenSort_step<<<blocks, threads>>>(dev_values, NUM_VALS);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    //MEM COPY FROM DEVICE TO HOST
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    cudaFree(dev_values);
}

double* CreateArray(int size) {

    double* arr = new double[size];
    for(int i = 0 ; i < size; ++i) {
        arr[i] = rand() % MAXBOUND;
    }
    return arr;
}


int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;
  SetSeed();

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;
  CALI_CXX_MARK_FUNCTION;
  CALI_MARK_BEGIN(main);

  double *values = (double*) malloc( NUM_VALS * sizeof(double));
  CALI_MARK_BEGIN(data_init);
  values = CreateArray(NUM_VALS);
  CALI_MARK_END(data_init);
  start = clock();
  oddEvenSort(values); /* Inplace */

  CALI_MARK_BEGIN(correctness_check);
  Correctness(values, NUM_VALS);
  CALI_MARK_END(correctness_check);
  stop = clock();
  CALI_MARK_END(main);


  // Store results in these variables.
    string algorithm, programmingModel,datatype, inputType, implementation_source;
    algorithm = "Bubble/Odd-Even Sort";
    programmingModel = "CUDA";
    datatype = "double but can change to float";
    inputType = "sorted";
    implementation_source = "All 3, Online, AI, and Handwritten";
    int sizeOfDatatype, inputSize, num_procs, num_threads, num_blocks, group_number;
    sizeOfDatatype = sizeof(double);
    inputSize = size_arr;
    num_procs = num_processors;
    num_threads = num_thre;
    num_blocks = 0;
    group_number = 20;
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
    adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}

/*
function oddEvenSort(arr):
    for p times - number of processors

        # i- Odd phase -start from taskid = 1 and depending if num of processes is even or odd then go up to the last process or last process - 1
            if taskid % 2 == 1:
                send array to taskid + 1
                receive array from taskid +1
            else:
                send array to taskid - 1
                receive array from taskid +1
            swap numbers if necessary 

        # Even phase -  even processes do this. start from taskid = 0 and end depending if num of proc is even or odd

            exchange array with assigned process.
            if you have to swap some numbers:


*/

// each process initializes their own part of the array and sorts
//each thread sorts their initial array

//when done, try to join arrays with another thread.
    
//because both arrays are sorted, comparison starts at the middle, and decreases by one after every loop. stop if already sorted

// 1 4 8 9 12     1 2 4 14

// 1 4 8 9 1 2 4 12 14
// 1 4 8 1 2 4 8 12 14

// when done, join with another thread.

//stop when all threads done ,slowly kill every


//return array


