#include <stdio.h>

// __global__ is a CUDA specifier that indicates a function that runs on the device and can be called from the host
__global__ void square(float *d_out, float *d_in) { 
    int idx = threadIdx.x; // threadIdx is a built-in variable that gives the index of the thread in the block.
    float f = d_in[idx]; // get the input value
    d_out[idx] = f * f; // square the input value and store it in the output array
}

int main(int argc, char const *argv[])
{
    const int ARRAY_SIZE = 64; // size of the array
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float); // size of the array in bytes

    // generate the input array on the host, h_ prefix for host memory
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i); // fill it with float values
    }
    float h_out[ARRAY_SIZE];

    // declare GPU memory pointers
    float *d_in; // gpu convention to declare pointers to device memory with d_ prefix
    float *d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES); // allocate memory on the device
    cudaMalloc((void **) &d_out, ARRAY_BYTES); // allocate memory on the device

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); // copy the input array from the host to the device

    square<<<1, ARRAY_SIZE>>>(d_out, d_in); // launch the kernel the characters <<< >>> are called execution configuration
    // each thread can have at most 1024 threads, so we can have 1024 threads per block. 
    // the first argument is the number of blocks, the second is the number of threads per block
    // scores are in d_out now, copy them back to the host. 


    // this is the same thing as saying 

    // square <<< dim3(1, 1, 1), dim3(ARRAY_SIZE, 1, 1), sharedMemSize >>> (d_out, d_in);

    // sharedMemSize is the number of bytes in shared memory that is dynamically allocated per block for this call in addition to the statically allocated memory.
    
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // there's also device to device memory copy

    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t" : "\n"); // four things per row
    }
    cudaFree(d_in); // free the memory on the device
    cudaFree(d_out); // free the memory on the device

    return 0;
}
