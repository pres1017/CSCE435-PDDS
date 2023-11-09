/*
*      ChatGPT was used to help with CUDA debugging. 
*
*
*/

#include <stdio.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* main = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* data_validation = "data_validation";

__global__ void merge(float *a, float *b, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int start = tid * n;
    int end = start + n;
    int i = start, j = start + n/2, k = start;

    while (i < start + n/2 && j < end) {
        if (a[i] <= a[j]) {
            b[k] = a[i];
            i++;
        } else {
            b[k] = a[j];
            j++;
        }
        k++;
    }
    while (i < start + n/2) {
        b[k] = a[i];
        i++;
        k++;
    }
    while (j < end) {
        b[k] = a[j];
        j++;
        k++;
    }
    for (int i = start; i < end; i++) {
        a[i] = b[i];
    }
}

void parallelMergeSort(float *a, float *b, int n) {
    int *dev_a, *dev_b;
    cudaMalloc((void**)&dev_a, n * sizeof(int));
    cudaMalloc((void**)&dev_b, n * sizeof(int));

    CALIPER_MARK_BEGIN(comm_large);
    cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    CALIPER_MARK_END(comm_large);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    CALIPER_MARK_BEGIN(comp);
    CALIPER_MARK_BEGIN(large_comp);
    for (int i = 2; i <= n; i *= 2) {
        for (int j = 0; j < n; j += i) {
            CALIPER_MARK_BEGIN(small_comp);
            merge<<<numBlocks, blockSize>>>(dev_a, dev_b, i);
            cudaDeviceSynchronize();
            CALIPER_MARK_END(small_comp);
        }
    }
    CALIPER_MARK_END(large_comp);
    CALIPER_MARK_END(comp);

    cudaMemcpy(a, dev_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
}

float randomFloat(int a, int b)
{
    if (a > b)
        return randomFloat(b, a);
    if (a == b)
        return a;
 
    return (float)randomInt(a, b) + randomFloat();
}

int main(int argc, char** argv) {
    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN(main);

    CALI_MARK_BEGIN(data_init);
    std::string arg = argv[1];
    const int n = std::stoi(arg);
    float *a = new float[n];

    for(int i = 0; i < n; i++){
        a[i] = randomFloat(1, 1000);
    }
    CALI_MARK_END(data_init);

    float *b = (float*)malloc(n * sizeof(float));

    parallelMergeSort(a, b, n);

    CALI_MARK_BEGIN(data_validation);
    printf("Sorted array: \n");
    for (int i = 1; i <= n; i++) {
        if(i != n && a[i] < a[i - 1]){
            printf("Not sorted\n");
            break;
        }
        printf("%d ", a[i - 1]);
    }
    printf("\n");
    CALI_MARK_END(data_validation);

    free(b);

    CALI_MARK_END(main);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", numBlocks); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();
}