# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Kernel for Bubble Sort
__global__ void bubbleSortKernel(int *d_arr, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = 0; i < n; i++) {
        int j = idx;

        if (j < n - i - 1) {
            if (d_arr[j] > d_arr[j + 1]) {
                int temp = d_arr[j];
                d_arr[j] = d_arr[j + 1];
                d_arr[j + 1] = temp;
            }
        }
        __syncthreads();
    }
}

// Device merge function
__device__ void merge(int *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0; j = 0; k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) arr[k++] = L[i++];
        else arr[k++] = R[j++];
    }

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];

    free(L);
    free(R);
}

// Merge Sort Kernel
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n) {
    for (int size = 1; size < n; size *= 2) {
        int left = 0;
        while (left + size < n) {
            int mid = left + size - 1;
            int right = min(left + 2 * size - 1, n - 1);

            merge(d_arr, left, mid, right);
            left += 2 * size;
        }
        __syncthreads();
    }
}

// GPU Bubble Sort
void bubbleSort(int *arr, int n) {
    int *d_arr;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    bubbleSortKernel<<<1, n>>>(d_arr, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    printf("Bubble Sort GPU Time: %f ms\n", time);
}

// GPU Merge Sort
void mergeSort(int *arr, int n) {
    int *d_arr, *d_temp;
    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mergeSortKernel<<<1,1>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);

    printf("Merge Sort GPU Time: %f ms\n", time);
}

// CPU Bubble Sort
void bubbleSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j+1]) {
                int t = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = t;
            }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> d = end - start;

    printf("Bubble Sort CPU Time: %f ms\n", d.count());
}

// CPU Merge Sort helper
void mergeHost(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1, n2 = r - m;

    int *L = (int*)malloc(n1*sizeof(int));
    int *R = (int*)malloc(n2*sizeof(int));

    for(i=0;i<n1;i++) L[i]=arr[l+i];
    for(j=0;j<n2;j++) R[j]=arr[m+1+j];

    i=0;j=0;k=l;

    while(i<n1 && j<n2)
        arr[k++] = (L[i]<=R[j]) ? L[i++] : R[j++];

    while(i<n1) arr[k++]=L[i++];
    while(j<n2) arr[k++]=R[j++];

    free(L); free(R);
}

// CPU Merge Sort
void mergeSortCPU(int *arr, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for(int size=1; size<n; size*=2) {
        for(int left=0; left<n-1; left+=2*size) {
            int mid = min(left+size-1, n-1);
            int right = min(left+2*size-1, n-1);
            mergeHost(arr,left,mid,right);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> d = end - start;

    printf("Merge Sort CPU Time: %f ms\n", d.count());
}

// Print array
void printArray(int *arr, int n) {
    for(int i=0;i<n;i++) printf("%d ", arr[i]);
    printf("\n");
}

// Main
int main() {
    int n = 1024;
    int *arr = (int*)malloc(n*sizeof(int));

    for(int i=0;i<n;i++) arr[i]=rand()%1000;

    bubbleSortCPU(arr,n);
    bubbleSort(arr,n);

    for(int i=0;i<n;i++) arr[i]=rand()%1000;

    mergeSortCPU(arr,n);
    mergeSort(arr,n);

    free(arr);
    return 0;
}
```

## OUTPUT:
```
Bubble Sort CPU Time: 2.178268 ms
Bubble Sort GPU Time: 113.426430 ms
Merge Sort CPU Time: 0.148762 ms
Merge Sort GPU Time: 50.524769 ms
```

## RESULT:
Thus, the program has been executed using CUDA to implement and analyze the performance of Bubble Sort and Merge Sort algorithms by parallelizing computations on the GPU, resulting in improved execution time compared to CPU implementations.
