# CSCE435 Project

**How we plan to communicate:** Discord

**Algorithms we plan to implement:** Bitonic Sort and Mergesort

**Bitonic Sort Pseudocode**

        for 2 to sizeBatch
                for j to 0
                    for 0 to sizeBatch
                        ixj = i power j;
                        if ixj > i {
                            if i & k == 0 && data[s + i] > data[s + ixj]
                                temp = d[s + i];
                                data[s + i] = data[s + ixj];
                                data[s + ixj] = temp;
                            if i & k != 0 && data[s + i] < data[s + ixj]
                                temp = data[s + i];
                                data[s + i] = data[s + ixj];
                                data[s + ixj] = temp;


    
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
