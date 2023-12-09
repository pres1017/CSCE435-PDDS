#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <mpi.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

/**********************
*
* Source Code: https://github.com/racorretjer/Parallel-Merge-Sort-with-MPI/blob/master/merge-mpi.c 
*
**********************/

const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comm_small = "comm_small";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* comp_small = "comp_small";

void merge(int *, int *, int, int, int);
void mergeSort(int *, int *, int, int);



int main(int argc, char** argv) {
  const char* main = "main";
  cali::ConfigManager mgr;
  mgr.start();
    
	CALI_MARK_BEGIN(main);

	int n = atoi(argv[1]);
	int *original_array = (int*)malloc(n * sizeof(int));
 
  int input = atoi(argv[2]);
  std::string type = "";  
  
  int numProcs = atoi(argv[3]);      
	
	int c;
	srand(time(NULL));
  CALI_MARK_BEGIN(data_init);
      if(input == 0 || input == 3){
        for (int i = 0; i < n; i++){
          original_array[i] = i;
        }
        type = "Sorted";
      }
        
      if(input == 1){
        for(int i = 0; i < n; i++){
          original_array[i] = rand() % 100000;
        }
        type = "Random";
      }
      
      if(input == 2){
        for(int i = 0; i < n; i++){
          original_array[i] = n - i;
        }
        type = "Reverse Sorted";
      }
      
      if(input == 3){
        for(int i = 0; i < n; i++){
          int chance = rand() % 100;
          int randIndex = rand() % n;
          if(chance <= 1){
            int tempVal = original_array[i];
            original_array[i] = original_array[randIndex];
            original_array[randIndex] = tempVal;
          }
          type = "1%perturbed";
        }
      }

  CALI_MARK_END(data_init);
  
	
	int world_rank;
	int world_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		
	int size = n/world_size;

	int *sub_array = (int*)malloc(size * sizeof(int));
  CALI_MARK_BEGIN(comm);
  CALI_MARK_BEGIN(comm_large);
  CALI_MARK_BEGIN(comm_small);
	MPI_Scatter(original_array, size, MPI_INT, sub_array, size, MPI_INT, 0, MPI_COMM_WORLD);
  CALI_MARK_END(comm_small);
  CALI_MARK_END(comm_large);
  CALI_MARK_END(comm);

  CALI_MARK_BEGIN(comp);
  CALI_MARK_BEGIN(comp_large);
	int *tmp_array = (int*)malloc(size * sizeof(int));
	mergeSort(sub_array, tmp_array, 0, (size - 1));
  CALI_MARK_END(comp_large);
  CALI_MARK_END(comp);

	int *sorted = NULL;
	if(world_rank == 0) {
		
		sorted = (int*)malloc(n * sizeof(int));
		
		}
	
	MPI_Gather(sub_array, size, MPI_INT, sorted, size, MPI_INT, 0, MPI_COMM_WORLD);
	
	if(world_rank == 0) {
		
		int *other_array = (int*)malloc(n * sizeof(int));
		mergeSort(sorted, other_array, 0, (n - 1));
    CALI_MARK_END(comp);
		
    bool isSorted = true;
    CALI_MARK_BEGIN(correctness_check);
		printf("This is the sorted array: ");
		for(c = 0; c < n - 1; c++) {
      if(original_array[c + 1] < original_array[c]){
        isSorted = false;
      }   
		}
   
    if(isSorted){
      printf("sorted");
    }else{
      printf("not sorted");
    }
	  CALI_MARK_END(correctness_check);
		printf("\n");
		printf("\n");
			
		free(sorted);
		free(other_array);
			
		}
	

	free(original_array);
	free(sub_array);
	free(tmp_array);

	MPI_Barrier(MPI_COMM_WORLD);
 
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
  adiak::value("num_procs", numProcs); // The number of processors (MPI ranks)
  adiak::value("num_threads", 0); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", 0); // The number of CUDA blocks 
  adiak::value("group_num", 20); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  
  CALI_MARK_END(main);

  mgr.stop();
  mgr.flush();

	MPI_Finalize();
	
	}

/********** Merge Function **********/
void merge(int *a, int *b, int l, int m, int r) {
  
	int h, i, j, k;
	h = l;
	i = l;
	j = m + 1;
	
	while((h <= m) && (j <= r)) {
		
		if(a[h] <= a[j]) {
			
			b[i] = a[h];
			h++;
			
			}
			
		else {
			
			b[i] = a[j];
			j++;
			
			}
			
		i++;
		
		}
		
	if(m < h) {
		
		for(k = j; k <= r; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	else {
		
		for(k = h; k <= m; k++) {
			
			b[i] = a[k];
			i++;
			
			}
			
		}
		
	for(k = l; k <= r; k++) {
		
		a[k] = b[k];
		
		}
   
	}

/********** Recursive Merge Function **********/
void mergeSort(int *a, int *b, int l, int r) {
	
	int m;
	
	if(l < r) {
		
		m = (l + r)/2;
		
		mergeSort(a, b, l, m);
		mergeSort(a, b, (m + 1), r);
		merge(a, b, l, m, r);
		
		}
		
	}
