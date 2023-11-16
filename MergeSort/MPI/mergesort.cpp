/*
*
* ChatGPT was used for help with MPI debugging. 
*
*/

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <string>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const char* whole = "whole";
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
void merge(int arr[], int l, int m, int r) {
    CALI_MARK_BEGIN(comp_small);
    int i, j, k;
    int n1 = m - l + 1;
    int n2 =  r - m;

    int L[n1], R[n2];

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
    CALI_MARK_END(comp_small);
}

// Function to perform merge sort
void mergeSort(int arr[], int l, int r) {
    CALI_MARK_BEGIN("comp_large");
    if (l < r) {
        int m = l+(r-l)/2;

        mergeSort(arr, l, m);
        mergeSort(arr, m+1, r);

        merge(arr, l, m, r);
    }
    CALI_MARK_END("comp_large");
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    CALI_MARK_BEGIN(whole);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string numVals = argv[1];
    const int n = std::stoi(numVals);
    
    std::string type = argv[2];
    const int input = std::stoi(type);
    
    std::string processes = argv[3];
    const int numProcesses = std::stoi(processes);
    

    int array[n];
    srand(time(NULL));
    
    if (rank == 0) {
    
      if(input == 0 || input == 3){
        for (int i = 0; i < n; i++){
          array[i] = i;
        }
        type = "Sorted";
      }
        
      if(input == 1){
        for(int i = 0; i < n; i++){
          array[i] = rand() % 100000;
        }
        type = "Random";
      }
      
      if(input == 2){
        for(int i = 0; i < n; i++){
          array[i] = n - i;
        }
        type = "Reverse Sorted";
      }
      
      if(input == 3){
        for(int i = 0; i < n; i++){
          int chance = rand() % 100;
          int randIndex = rand() % n;
          if(chance <= 1){
            int tempVal = array[i];
            array[i] = array[randIndex];
            array[randIndex] = array[i];
          }
          type = "1%perturbed";
        }
      }
      /*
      for (int i = 0; i < n; i++) {
          std::cout << array[i] << " ";
      }
      std::cout << std::endl;
      */
    }

    cali::ConfigManager mgr;
    mgr.start();

    MPI_Bcast(array, n, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = n / size;
    int local_array[local_n];

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Scatter(array, local_n, MPI_INT, local_array, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    mergeSort(local_array, 0, local_n - 1);
    CALI_MARK_END(comp);

    MPI_Gather(local_array, local_n, MPI_INT, array, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    CALI_MARK_BEGIN(data_validation);
    if (rank == 0) {
        mergeSort(array, 0, n - 1);
        std::cout << "Sorted array: ";
        for (int i = 1; i <= n; i++) {
            if(i != n && array[i] < array[i - 1]){
                printf("Not sorted\n");
                break;
            }
            // std::cout << array[i - 1] << " ";
        }
        std::cout << std::endl;
    }
    CALI_MARK_END(data_validation);

    CALI_MARK_END(whole);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(int)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", type); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}
