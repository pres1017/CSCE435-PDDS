#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <float.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

const int THRESHOLD = 1024;

#define MASTER 0               
#define FROM_MASTER 1          
#define FROM_WORKER 2          

int NUM_VALS;
int sorting;


float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}


void bitonic_sort(float *d, int s, int sizeBatch, int elementPerProcess) {
    int i, j, k;
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    for (k = 2; k <= sizeBatch; k = 2 * k) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            for (i = 0; i < sizeBatch; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && d[s + i] > d[s + ixj]) {
                        float temp = d[s + i];
                        d[s + i] = d[s + ixj];
                        d[s + ixj] = temp;
                    }
                    if ((i & k) != 0 && d[s + i] < d[s + ixj]) {
                        float temp = d[s + i];
                        d[s + i] = d[s + ixj];
                        d[s + ixj] = temp;
                    }
                }
            }
        }
    }
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");
}

void data_init(float *arr, int length, int sorting)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i) {
      arr[i] = random_float();
    }
}


void sort_splitters(float *s, int num_s) {
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    
    for (int i = 0; i < num_s - 1; i++) {
        int min = i;
        for (int j = i + 1; j < num_s; j++) {
            if (s[j] < s[min]) {
                min = j;
            }
        }
        if (min != i) {
            float temp = s[i];
            s[i] = s[min];
            s[min] = temp;
        }
    }

    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");
}

void exchange(float *d, int sizeBatch, int partner) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    float *partner_data = (float*)malloc(sizeof(float) * sizeBatch / 2);

    if (partner_data == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return;
    }

    MPI_Status status;

    if (rank < partner) {
        CALI_MARK_BEGIN("comm_large");
        MPI_Send(d + sizeBatch / 2, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
        MPI_Recv(partner_data, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &status);
        memcpy(d + sizeBatch / 2, partner_data, sizeof(float) * sizeBatch / 2);
        CALI_MARK_END("comm_large");
    } else {
        CALI_MARK_BEGIN("comm_small");
        MPI_Recv(partner_data, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD, &status);
        MPI_Send(d, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
        memcpy(d, partner_data, sizeof(float) * sizeBatch / 2);
        CALI_MARK_END("comm_small");
    }

    free(partner_data);
}



int is_sorted(float *d, int elements) {
    for (int i = 0; i < elements - 1; ++i) {
        if (d[i] > d[i + 1]) {
            return 0; 
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int	numtasks,             
        taskid;               
    float *global_array = NULL;
    int sizeOfArray;


    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);


    sizeOfArray = atoi(argv[1]);
    NUM_VALS = sizeOfArray;

    cali::ConfigManager mgr;
    mgr.start();

    CALI_MARK_BEGIN("data_init");
    global_array = (float *)malloc(sizeOfArray * sizeof(float));
    if (global_array == NULL) {
        printf("Cannot allocate enough memory for the global array.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    if (taskid == MASTER) {
        data_init(global_array, NUM_VALS, 0);
    }
    CALI_MARK_END("data_init");


    int elementsPerProcess = NUM_VALS / numtasks;

    if(elementsPerProcess > 1024000000000000){
        sort_splitters(global_array, elementsPerProcess);
    } else {
        bitonic_sort(global_array, taskid * elementsPerProcess, elementsPerProcess, elementsPerProcess);
    }


    int rank_partner = taskid ^ 1; 
    exchange(global_array, elementsPerProcess, rank_partner);


    CALI_MARK_BEGIN("correctness_check");
    int sorted = is_sorted(global_array, NUM_VALS);
    CALI_MARK_END("correctness_check");

    if (taskid == 0) {
        if (sorted) {
            printf("Data is sorted correctly.");
        } else {
            printf("Data is not sorted correctly.");
        }
    }

    free(global_array);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();


    const char *algorithm = "BitonicSort";
    const char *programmingModel = "MPI";
    const char *datatype = "float";
    int sizeOfDatatype = sizeof(float);
    int inputSize = NUM_VALS; 
    const char *inputType = "Random";
    int num_procs, num_threads = 1, num_blocks = 1;
    int group_number = 20;
    const char *implementation_source = "Online";


    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", inputSize);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", numtasks);
    adiak::value("num_threads", 0);
    adiak::value("num_blocks", 0);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    mgr.stop();
    mgr.flush();
    MPI_Finalize();
}