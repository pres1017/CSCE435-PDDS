# CSCE435 Project

**How we plan to communicate:** Discord

**Algorithms we plan to implement:** Bitonic Sort and Mergesort

**Bitonic Sort Pseudocode**

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


    
**Mergesort Pseudocode**

MergeSort(arr, left, right):

    if left > right 
    
        return
        
    mid = (left+right)/2
    
    mergeSort(arr, left, mid)
    
    mergeSort(arr, mid+1, right)
    
    merge(arr, left, mid, right)
    
end


**What versions we plan to compare:** OpenMP and MPI
