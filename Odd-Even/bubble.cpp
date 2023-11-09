//split the array by len(array) / numthreads. <-- this the size that each of the array that each thread is getting
#include <iostream>
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

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


using std::cout, std::endl;


void SetSeed(int rank) {
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr)) + rank;
    std::srand(seed);
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

void PrintArr(double* arr, int size) {
    for(int i = 0; i < size; ++i) {
        cout << arr[i] << " ";
    }
    cout << " " << endl;
}


int main (int argc, char *argv[])
{
    CALI_CXX_MARK_FUNCTION;
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

    local_arr = CreateArray(size_local_arr);

    // #####    sort local_array
    std::sort(local_arr, local_arr + size_local_arr);
    holder = new double[size_local_arr];
    //MPI_Gather(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, global_array[offset], size_local_arr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    int phase = 1; // 1 is odd, -1 is even
   // for p times - number of processors
    int send_rec_from;

    for(int i = 0; i < num_processors; ++i) {
        //include last processes, exlude first process
        if (taskid % 2 == 1 && num_processors % 2 == 1 ) { //<-- take into consideration if even or odd num of processors
            if(phase == 1 ) { //in odd phase - include last processes, 
                send_rec_from = taskid + 1
                MPI_Send(local_arr, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD);
                MPI_Status status;

                MPI_Recv(holder, size_local_arr * sizeof(double), MPI_DOUBLE, send_rec_from, 0, MPI_COMM_WORLD, &status);
                //mpi_receive(send_rec_from)
                //call swap function
                SwapLower(local_arr, holder, size_local_arr);


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

        /*##### dont include below
        # i- Odd phase -start from taskid = 1 and depending if num of processes is even or odd then go up to the last process or last process - 1
            if taskid % 2 == 1:
                send_rec_from = taskid - 1
                send array to taskid + 1
                receive array from taskid +1
            else:
                send array to taskid - 1
                receive array from taskid +1
            swap numbers if necessary 

        # Even phase -  even processes do this. start from taskid = 0 and end depending if num of proc is even or odd

            exchange array with assigned process.
            if you have to swap some numbers:

        ##### */
    }
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
    delete[] arr
    return 0;
}

/*


40 size    7

10 10 
20 

10 10
20

50 


notes:



*/
