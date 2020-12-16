#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
{
    __shared__ bench_t A_tile[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ bench_t B_tile[BLOCK_SIZE*BLOCK_SIZE];

    
    unsigned int i = blockIdx.x *  BLOCK_SIZE + threadIdx.x;
    unsigned int j = blockIdx.y *  BLOCK_SIZE + threadIdx.y;
    
    
    bench_t acumulated = 0;
    unsigned int idx = 0;
    // load memory
    for (unsigned int sub = 0; sub < gridDim.x; ++sub)
    {
        
        idx = i * n + sub * BLOCK_SIZE + threadIdx.y;

        if(idx >= m*n)
        {
            A_tile[threadIdx.x * BLOCK_SIZE+ threadIdx.y] = 0;
        }
        else
        {   
            A_tile[threadIdx.x * BLOCK_SIZE + threadIdx.y] = A[idx];
        }
        idx = (sub * BLOCK_SIZE + threadIdx.x) * w + j;

        if (idx >= m*w)
        {
            B_tile[threadIdx.x * BLOCK_SIZE +  threadIdx.y] = 0;
        }
        else
        {   
            B_tile[threadIdx.x* BLOCK_SIZE + threadIdx.y] = B[idx];
        }
        __syncthreads();
        for (unsigned int k = 0; k < BLOCK_SIZE; ++k)
        {
            acumulated +=  A_tile[threadIdx.x*BLOCK_SIZE + k] * B_tile[k*BLOCK_SIZE + threadIdx.y];
        }
        __syncthreads();

    }
    if (i < n && j < w)
    {
        
        C[i *n + j] = acumulated;
    }
}

void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new cudaEvent_t;
    device_object->stop = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start);
    cudaEventCreate(device_object->stop);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){

   // Allocate the device input vector A
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device output vector C
    err = cudaMalloc((void **)&device_object->d_C, size_c_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->d_B, h_B, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*device_object->stop_memory_copy_device);
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(ceil(float(n)/dimBlock.x), ceil(float(m)/dimBlock.y));
    cudaEventRecord(*device_object->start);
    matrix_multiplication_kernel<<<dimGrid, dimBlock>>>(device_object->d_A, device_object->d_B, device_object->d_C, n, m, w);
    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    cudaEventRecord(*device_object->start_memory_copy_host);
	cudaMemcpy(h_C, device_object->d_C, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
	}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f miliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
	cudaError_t err = cudaSuccess;
	err = cudaFree(device_object->d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }


    // delete events
    delete device_object->start;
    delete device_object->stop;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
}
