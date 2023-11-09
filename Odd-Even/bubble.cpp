//split the array by len(array) / numthreads. <-- this the size that each of the array that each thread is getting
#include <iostream>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

#define MAXBOUND 1000001  // up to a million


const char* main = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";


using std::cout, std::endl, std::string;


void SetSeed(int rank) {
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) + rank;
    std::srand(seed);
}
void Correctness(double* arr, int size) {
    for(int i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            cout << "Not sorted" << endl;
            return;
        }
    }
    cout << "Sorted" << endl;
}

double* OddEvenSort(int size, double** arr) {
    
    return arr;
    
}

double* CreateArray(int size) {

    double* arr = new double[size];
    for(int i = 0 ; i < size; ++i) {
        arr[i] = rand() % MAXBOUND;
    }
    return arr;
}

void SwapLower(double* &local, double* &holder,int size) {
    for (int i = 0; i < size; ++i) {
        if (holder[i] < local[i]) {
            local[i] = holder[i];
        }
    }
}   

void SwapHigher(double* &local, double* &holder,int size) {
    for (int i = 0; i < size; ++i) {
        if (holder[i] > local[i]) {
            local[i] = holder[i];
        }
    }
}   
void PrintArr(double* arr, int size) {
    for(int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << " " << endl;
}


int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    CALI_MARK_BEGIN(main);

    int taskid,numtasks,rc;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
    }
    numworkers = numtasks-1;

    // WHOLE PROGRAM COMPUTATION PART STARTS HERE
    //CALI_MARK_BEGIN(whole_computation);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    //start
    SetSeed(taskid);
    double* arr;
    double* local_arr;
    double* holder;
    // need to change
    int num_processors = atoi(argv[2]);
    int size_arr = atoi(argv[1]);

    int size_local_arr = size_arr / num_processors;
    int offset = taskid * size_local_arr;
    if (taskid == MASTER) {
        // allocating memory only for master process
        arr = new double[size_arr];
        // for(int i = 0; i < size_arr; ++i){
        //     arr[i] = new double;
        // }
    }
    CALI_MARK_BEGIN(data_init);
    local_arr = CreateArray(size_local_arr);
    CALI_MARK_END(data_init);

    // #####    sort local_array
    holder = new double[size_local_arr];
    //MPI_Gather(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, global_array[offset], size_local_arr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int phase = 1; // 1 is odd, -1 is even
   // for p times - number of processors
    int send_rec_from;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    std::sort(local_arr, local_arr + size_local_arr);
    CALI_MARK_BEGIN(comm);
    for(int i = 0; i < num_processors; ++i) {
        MPI_Status status;

        //include last processes, exlude first process
        if (taskid % 2 == 1 && num_processors % 2 == 1 ) { //<-- take into consideration if even or odd num of processors
            if(phase == 1 ) { //in odd phase - include last processes, 
                send_rec_from = taskid + 1
                CALI_MARK_BEGIN(comm_small);
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                CALI_MARK_END(comm_small);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp_small);
                SwapLower(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);

            }
            else if (phase == -1 && taskid != numtasks - 1) {//in even phase - exclude last
                send_rec_from = taskid - 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapHigher(local_arr, holder, size_local_arr);

            }
        }
        // first and last processes do not get a pair
        else if(taskid % 2 == 1 && num_processors % 2 == 0) {

            if (phase == 1 && rank != numtasks - 1) { //in odd phase - exclude last
                send_rec_from = taskid + 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapLower(local_arr, holder, size_local_arr);
            }
            else if (phase == -1) { //in even phase - include everyone
                send_rec_from = taskid - 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapHigher(local_arr, holder, size_local_arr);
            }
        }
        //last process does not get a pair
        else if(taskid % 2 == 0 && num_processors % 2 == 1) {

            if (phase == 1 && taskid != 0) { //in odd phase - exclude first
                send_rec_from = taskid - 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapHigher(local_arr, holder, size_local_arr);
            }
            else if (phase == -1 && taskid != numtasks - 1) { //in even phase - exclude last
                send_rec_from = taskid + 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapLower(local_arr, holder, size_local_arr);
            }
        }
        // they all get a pair - even and even
        else { // taskid % 2 == 0 and num_processors % 2 == 0   <--- take into consideration num of processes. even or odd
            if (phase == 1 && taskid != 0) { //in odd phase - exclude first and last
                send_rec_from = taskid - 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapHigher(local_arr, holder, size_local_arr);
            }
            else if (phase == -1) { //in even phase - include everyone
                send_rec_from = taskid + 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                SwapLower(local_arr, holder, size_local_arr);
            }
        }
        phase = -1 * phase; // change between odd odd and even phase
    }
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    CALI_MARK_START(comm_large);
    MPI_Gather(local_arr, size_local_arr, MPI_DOUBLE, arr[offset], size_local_arr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);
    if (taskid == 0){
        MPI_Gather(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, arr[offset], size_local_arr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        PrintArr(arr, size_arr);
        delete[] arr;
        //delete[] local_arr;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    CALI_MARK_START(correctness_check);
    Correctness(arr, size_arr);
    CALI_MARK_END(correctness_check);
    CALI_MARK_END(main);
    delete[] arr;
    
    if (taskid == 0) {
        string algorithm, programmingModel,datatype, inputType, implementation_source;
        algorithm = "Bubble/Odd-Even Sort";
        programmingModel = "MPI";
        datatype = "double but can change to float";
        inputType = "sorted";
        implementation_source = "All 3, Online, AI, and Handwritten";
        int sizeOfDatatype, inputSize, num_procs, num_threads, num_blocks, group_number;
        sizeOfDatatype = sizeof(double);
        inputSize = size_arr;
        num_procs = num_processors;
        num_threads = 0;
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
    }
    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}
