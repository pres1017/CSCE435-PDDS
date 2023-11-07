#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <caliper/cali.h>
#include <adiak.hpp>

const int THRESHOLD = 1024;

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

    if (rank == partner) {
        memcpy(partner_data, d + (rank < partner ? 0 : sizeBatch / 2), sizeof(float) * sizeBatch / 2);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (rank < partner) {
        CALI_MARK_BEGIN("comm_large");
        MPI_Send(d + sizeBatch / 2, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(partner_data, sizeBatch / 2, MPI_FLOAT, partner, MPI_COMM_WORLD);

    if (rank >= partner) {
        CALI_MARK_BEGIN("comm_small");
        MPI_Send(d, sizeBatch / 2, MPI_FLOAT, partner, 0, MPI_COMM_WORLD);
    }

    if (rank < partner) {
        memcpy(d + sizeBatch / 2, partner_data, sizeof(float) * sizeBatch / 2);
        CALI_MARK_END("comm_large");
    } else {
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
    MPI_Init(&argc, &argv);

    adiak::init(NULL);
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();


    const char *algorithm = "BitonicSort";
    const char *programmingModel = "MPI";
    const char *datatype = "float";
    int sizeOfDatatype = sizeof(float);
    int inputSize = 1024; 
    const char *inputType = "Random";
    int num_procs, num_threads = 1, num_blocks = 1;
    int group_number = 1;
    const char *implementation_source = "Handwritten";

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);


    adiak::value("Algorithm", algorithm);
    adiak::value("ProgrammingModel", programmingModel);
    adiak::value("Datatype", datatype);
    adiak::value("SizeOfDatatype", sizeOfDatatype);
    adiak::value("InputSize", inputSize);
    adiak::value("InputType", inputType);
    adiak::value("num_procs", num_procs);
    adiak::value("num_threads", num_threads);
    adiak::value("num_blocks", num_blocks);
    adiak::value("group_num", group_number);
    adiak::value("implementation_source", implementation_source);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CALI_MARK_FUNCTION_BEGIN("main");

    CALI_MARK_BEGIN("data_init");
    float *data = (float *)malloc(sizeof(float) * inputSize);
    if (data == NULL) {
        fprintf(stderr, "Memory allocation failed");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < inputSize; ++i) {
        data[i] = (float)rand() / RAND_MAX; 
    }
    CALI_MARK_END("data_init");


    int elementsPerProcess = inputSize / num_procs;

    if(elementsPerProcess < 1024){
        sort_splitters(data, elementsPerProcess);
    } else {
        bitonic_sort(data, rank * elementsPerProcess, elementsPerProcess, elementsPerProcess);
    }
    


    int rank_partner = rank ^ 1; 
    exchange(data, elementsPerProcess, rank_partner);


    CALI_MARK_BEGIN("correctness_check");
    int sorted = is_sorted(data, inputSize);
    CALI_MARK_END("correctness_check");

    if (rank == 0) {
        if (sorted) {
            printf("Data is sorted correctly.");
        } else {
            printf("Data is not sorted correctly.");
        }
    }

    free(data);
    CALI_MARK_FUNCTION_END();
    

    MPI_Finalize();
    return 0;
}

