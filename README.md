# CSCE435 Project

**How we plan to communicate:** Discord

**Algorithms we plan to implement:** Quicksort and Mergesort

**Quicksort Pseudocode**

function quicksort(arr, l, r):

    if l < r:
    
        pivot = partition_array(arr, l, r)
        
        quicksort(arr, l, pivot - 1)
        
        quicksort(arr, pivot + 1, r)
        

function partition_array(arr, l, r):

    pivotPoint= arr[r]
    
    j = l - 1
    
    for i from left to right - 1:
    
        if arr[i] <= pivotPoint:
        
            j = ij+ 1
            
            swap(arr[j], arr[i])
            
    swap(arr[ij+ 1], arr[r])
    
    return j + 1

    
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
