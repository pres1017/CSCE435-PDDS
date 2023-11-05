/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int NUM_VALS;
/* Define Caliper region names */
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

bool correctness_check(float *arr, int length){
  bool sorted = true;
  for (int i = 0; i < length-1; ++i) {
    if(arr[i]>arr[i+1]){
      sorted = false;
      break;
    }
  }
  return sorted;
}

void parallelEnumerationSort(float *array, int size, int rank, int numprocs) {

    // Broadcast the entire array to all processes
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Bcast(array, size, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    int local_size = size / numprocs;
    int start = rank * local_size;
    int end = (rank + 1) * local_size;//(rank == numprocs - 1) ? size : start + local_size;
    int *local_sorted_indices = (int *)malloc(size/numprocs * sizeof(int));

    // Perform enumeration sort on the entire array for the assigned range
    for (int i = start; i < end; ++i) {
        int count = 0;
        for (int j = 0; j < size; ++j) {
            if (array[i] > array[j] || (array[i] == array[j] && i > j)) {
                count++;
            }
        }
        local_sorted_indices[i-start] = count; // Place in the correct position
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    int *sorted_indices = NULL;
    // Gather the sorted segments into the global sorted array
    if (rank == MASTER) {
        sorted_indices = (int *)malloc(size * sizeof(int));
    }
    MPI_Gather(local_sorted_indices, local_size, MPI_INT, sorted_indices, local_size, MPI_INT, MASTER, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    // Merge the sorted segments in the master
    if (rank == MASTER) {
        float *array_cpy = (float *)malloc(size * sizeof(float));
        for(int i = 0; i < size; i++){
            array_cpy[i] = array[i];
        }
        
        for(int i = 0; i < size; i++){
            array[sorted_indices[i]] = array_cpy[i];
        }
    }
}
// no need for MPI_Barrier because Bcast and Gather synchronize all processes implicitly

int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;

    int	numtasks,              /* number of tasks */
        taskid;                /* a task identifier */
    float *global_array = NULL;
    int sizeOfArray;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    if (argc != 2) {
        if (taskid == MASTER) {
            printf("\n Please provide the size of the arrray to be sorted\n");
        }
        MPI_Finalize();
        exit(1);
    }

    sizeOfArray = atoi(argv[1]);
    NUM_VALS = sizeOfArray;

    cali::ConfigManager mgr;
    mgr.start();

    // Master initializes the array
    global_array = (float *)malloc(sizeOfArray * sizeof(float));
    if (global_array == NULL) {
        printf("Cannot allocate enough memory for the global array.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    // fill the initial global array with random values
    if (taskid == MASTER) {
        // Initialize the array with random values
        array_fill(global_array, NUM_VALS);
        // printf("Random Array:\n");
        // for (int i = 0; i < sizeOfArray; i++) {
        //     printf("%f ", global_array[i]);
        // }
        // printf("\n");
    }

    // Perform the parallel enumeration sort
    parallelEnumerationSort(global_array, sizeOfArray, taskid, numtasks);

    if (taskid == MASTER) {
        // Print the sorted array
        // printf("Sorted Array:\n");
        // for (int i = 0; i < sizeOfArray; i++) {
        //     printf("%f ", global_array[i]);
        // }
        bool sorted = correctness_check(global_array, NUM_VALS);
        printf("\nSorted is: %s\n", sorted ? "true" : "false");
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();
   // MPI_Comm_free(&MPI_COMM_WORKER);
   MPI_Finalize();
}