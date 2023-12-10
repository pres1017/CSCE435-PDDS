//split the array by len(array) / numthreads. <-- this the size that each of the array that each thread is getting
// node -> 48 cores - > 2 threads per core      96 threads per node
#include <iostream>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string>

#include <random>

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


const char* mainC = "main";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";
const char* MPI_SendC = "MPI_Send";
const char* MPI_RecvC = "MPI_Recv";

//const char* SwapLower = "SwapLower";
//const char* SwapHigher = "SwapHigher";
using std::cout;
using std::endl;
using std::string;


void SetSeed(int rank) {
    std::random_device rd;
    unsigned int seed = rd() + static_cast<unsigned int>(rank);
    // Seed the random number generator with the combined seed
    //srand(seed);
    //unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) + rank;
    std::srand(seed);
}

float random_float()
{
  int num = rand() % MAXBOUND;
  return (float)num;
}
void Correctness(float* arr, int size) {
    for(int i = 0; i < size - 1; ++i) {
        if (arr[i] > arr[i + 1]) {
            cout << "Not sorted" << endl;
            return;
        }
    }
    cout << "Sorted" << endl;
}

float* OddEvenSort(int size, float* arr) {
    
    return arr;
    
}

float* CreateArray(int size, int type) {
    float* arr = new float[size];
    if (type == 0) {
        float step = 1.0;
        for(int i = 0; i < size; ++i) {
            arr[i] =  step*i;
        }
    }
    else if(type == 1) {
        for(int i = 0 ; i < size; ++i) {
            arr[i] = random_float();
        }
    }
    else if (type == 2) {
        float step = 1.0;
        for(int i = size-1; i >= 0; --i) {
            arr[i] = i * step;
        }
    }
    else if (type == 3) {
        float step = 1.0; // Step size for the sorted array

        // Initialize the array with sorted values
        for (int i = 0; i < size; ++i) {
            arr[i] = i * step;
        }

        // Calculate 1% of the array's length
        int perturbCount = size / 100;

        for (int i = 0; i < perturbCount; i++) {
            // Select two random indices
            int idx1 = rand() % size;
            int idx2 = rand() % size;

            // Swap the elements at these indices
            float temp = arr[idx1];
            arr[idx1] = arr[idx2];
            arr[idx2] = temp;
        }
    }
    return arr;
}

void SwapLower(float* &local, float* &holder,int size) {
    float* temp = new float[size * 2];
    int a = 0;
    int b = 0;
    int c = 0;
    while(a < size && b < size) {
        if(holder[b] < local[a]) {
            temp[c] = holder[b];
            ++c;
            ++b;
        }
        else {
            temp[c] = local[a];
            ++c;
            ++a;
        }
    }

    while (a < size) {
        temp[c] = local[a];
        c++;
        a++;
    }
    
    while (b < size) {
        temp[c] = holder[b];
        c++;
        b++;
    }
    for(int i = 0; i < size; ++i) {
        local[i] = temp[i];
    }
    delete[] temp;
}   


void SwapHigher(float* &local, float* &holder,int size) {
    float* temp = new float[size * 2];
    int a = 0;
    int b = 0;
    int c = 0;
    while(a < size && b < size) {
        if(holder[b] < local[a]) {
            temp[c] = holder[b];
            ++c;
            ++b;
        }
        else {
            temp[c] = local[a];
            ++c;
            ++a;
        }
    }
    while (a < size) {
        temp[c] = local[a];
        c++;
        a++;
    }
    
    while (b < size) {
        temp[c] = holder[b];
        c++;
        b++;
    }
    for(int i = 0; i < size; ++i) {
        local[i] = temp[i + size];
    }
    delete[] temp;
}


void PrintArr(float* arr, int size) {
    for(int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << " " << endl;
}


int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
    //CALI_MARK_BEGIN(mainC);
    MPI_Init(&argc,&argv);
    double starter = MPI_Wtime();

    int taskid,numtasks,rc;
    double min_time, max_time, avg_time, total_time, var_time = 0;
    double timer = 0;
    
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
    }
    int numworkers = numtasks-1;
    // WHOLE PROGRAM COMPUTATION PART STARTS HERE
    //CALI_MARK_BEGIN(whole_computation);

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();

    //start
    SetSeed(taskid);
    float* arr;
    float* local_arr;
    float* holder;
    // need to change

    int num_processors = numtasks; //atoi(argv[2]);
    int size_arr = atoi(argv[1]);
    //printf("%s", argv[2]);
    int size_local_arr = size_arr / num_processors;
    int offset = taskid * size_local_arr;
    if (taskid == MASTER) {
        // allocating memory only for master process
        arr = new float[size_arr];
    }
    CALI_MARK_BEGIN(data_init);
    local_arr = CreateArray(size_local_arr, atoi(argv[2]));
    //PrintArr(local_arr, size_local_arr);
    CALI_MARK_END(data_init);

    holder = new float[size_local_arr];
    int phase = 1; // 1 is odd, -1 is even
   // for p times - number of processors
    int send_rec_from;
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    std::sort(local_arr, local_arr + size_local_arr);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
    //CALI_MARK_BEGIN(comm);
    //cout << "working" << endl;
    for(int i = 0; i < num_processors; ++i) {
        MPI_Status status;
        //printf("iteration %d from task %d \n", i, taskid);
        //PrintArr(local_arr, size_local_arr);
        //include last processes, exlude first process
        if (taskid % 2 == 1 && num_processors % 2 == 1 ) { //<-- take into consideration if even or odd num of processors
            if(phase == 1 ) { //in odd phase - include last processes, 
                send_rec_from = taskid + 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr, MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);
                MPI_Recv(holder, size_local_arr, MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapLower(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
            else if (phase == -1 && taskid != numtasks - 1) {//in even phase - exclude last
                send_rec_from = taskid - 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);

                MPI_Recv(holder, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapHigher(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);

            }
        }
        // first and last processes do not get a pair
        else if(taskid % 2 == 1 && num_processors % 2 == 0) {

            if (phase == 1 && taskid != numtasks - 1) { //in odd phase - exclude last
                send_rec_from = taskid + 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);
                MPI_Recv(holder, size_local_arr, MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapLower(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
            else if (phase == -1) { //in even phase - include everyone
                send_rec_from = taskid - 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr, MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);

                MPI_Recv(holder, size_local_arr, MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapHigher(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
        }
        //last process does not get a pair
        else if(taskid % 2 == 0 && num_processors % 2 == 1) {

            if (phase == 1 && taskid != 0) { //in odd phase - exclude first
                send_rec_from = taskid - 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);

                MPI_Recv(holder, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapHigher(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
            else if (phase == -1 && taskid != numtasks - 1) { //in even phase - exclude last
                send_rec_from = taskid + 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);

                MPI_Recv(holder, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapLower(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
        }
        // they all get a pair - even and even
        else { // taskid % 2 == 0 and num_processors % 2 == 0   <--- take into consideration num of processes. even or odd
            if (phase == 1 && taskid != 0) { //in odd phase - exclude first and last
                send_rec_from = taskid - 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);

                MPI_Recv(holder, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapHigher(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
            else if (phase == -1) { //in even phase - include everyone
                send_rec_from = taskid + 1;
                CALI_MARK_BEGIN(comm);
                CALI_MARK_BEGIN(comm_small);
                CALI_MARK_BEGIN(MPI_SendC);
                MPI_Send(local_arr, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD);
                //MPI_Status status;
                CALI_MARK_END(MPI_SendC);

                CALI_MARK_BEGIN(MPI_RecvC);
                MPI_Recv(holder, size_local_arr , MPI_FLOAT, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //swap values
                CALI_MARK_END(MPI_RecvC);
                CALI_MARK_END(comm_small);
                CALI_MARK_END(comm);
                //mpi_receive(send_rec_from)
                //call swap function
                CALI_MARK_BEGIN(comp);
                CALI_MARK_BEGIN(comp_small);
                SwapLower(local_arr, holder, size_local_arr);
                CALI_MARK_END(comp_small);
                CALI_MARK_END(comp);
            }
        }
        phase = -1 * phase; // change between odd odd and even phase
    }
    //CALI_MARK_END(comp_large);
    //CALI_MARK_END(comp);
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    MPI_Gather(local_arr, size_local_arr, MPI_FLOAT, &arr[offset], size_local_arr, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    //wait for all processes to finish
    //MPI_Barrier(MPI_COMM_WORLD);
    timer = MPI_Wtime() - starter;
    if (taskid == 0){
        //MPI_Gather(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, arr[offset], size_local_arr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //PrintArr(arr, size_arr);
        CALI_MARK_BEGIN(correctness_check);
        Correctness(arr, size_arr);
        CALI_MARK_END(correctness_check);
        delete[] arr;
        //delete[] local_arr;
    }
    //MPI_Barrier(MPI_COMM_WORLD);
    
    //CALI_MARK_END(mainC);

    //CALI_MARK_END(comm);
    delete[] holder;
    delete[] local_arr;
    //delete[] arr;
    MPI_Reduce (&timer,&min_time,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
    MPI_Reduce (&timer,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce (&timer,&total_time,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    
    if (taskid == 0) {
        avg_time = total_time / numtasks;
    }
    //MPI_Scatter(&avg_time, 1, MPI_DOUBLE,&avg, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    MPI_Bcast(&avg_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double myvar = pow(timer - avg_time,2);
    //cout << myvar << endl;

    MPI_Reduce (&myvar,&var_time,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); //variance sum
    
    if (taskid == 0) {
        string algorithm, programmingModel,datatype, inputType, implementation_source;
        algorithm = "Bubble/Odd-Even Sort";
        programmingModel = "MPI";
        datatype = "float";
        if(atoi(argv[2]) == 0) {
            inputType = "Sorted";
        }
        else if (atoi(argv[2]) == 1) {
            inputType = "Random";
        }
        else if (atoi(argv[2]) == 2) {
            inputType = " Reverse sorted";
        }
        else{
            inputType = "1%perturbed";
        }
        implementation_source = "All 3, Online, AI, and Handwritten";
        int sizeOfDatatype, inputSize, num_procs, num_threads, num_blocks, group_number;
        sizeOfDatatype = sizeof(float);
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

        //MPI_Recv(&min_time, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        
        
        var_time = var_time / numtasks;

        adiak::value("min", min_time); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("max", max_time); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("average", avg_time); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("variance", var_time); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("total", total_time); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        //cout << min_time << " " << max_time << " " << avg_time << " " << var_time << " " << total_time << endl;

        adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
        adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
        adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
        adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
        adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", implementation_source); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    }
    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
    return 0;
}
