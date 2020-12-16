#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

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


bool device_memory_init(GraficObject *device_object, int64_t size_b_matrix){
    cudaError_t err = cudaSuccess;
    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, (size_b_matrix /2) * sizeof(bench_cuda_complex));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size){
    cudaError_t err = cudaSuccess;
    bench_cuda_complex *h_signal = (bench_cuda_complex *)malloc(sizeof(bench_cuda_complex) * (size/2));
    for (unsigned int i = 0; i < (size/2); ++i){
        h_signal[i].x = h_B[i * 2];
        h_signal[i].y = h_B[i * 2 + 1];
    }

    cudaEventRecord(*device_object->start_memory_copy_device);
    err = cudaMemcpy(device_object->d_B, h_signal, sizeof(bench_cuda_complex) * (size / 2), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, int64_t size){

    cufftHandle plan;

    cudaEventRecord(*device_object->start);
    
    #ifdef FLOAT
    cufftPlan1d(&plan, size, CUFFT_C2C, 1);
    cufftExecC2C(plan, (cufftComplex *)device_object->d_B, (cufftComplex *)device_object->d_B, CUFFT_FORWARD);
    #else 
    cufftPlan1d(&plan, size, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, (cufftDoubleComplex *)device_object->d_B, (cufftDoubleComplex *)device_object->d_B, CUFFT_FORWARD);
    #endif
    
    cudaEventRecord(*device_object->stop);
    cufftDestroy(plan);
    
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size){
    bench_cuda_complex *h_signal = (bench_cuda_complex *)malloc(sizeof(bench_cuda_complex) * (size/2));
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_signal, device_object->d_B, (size/2) * sizeof(bench_cuda_complex), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);

    for (unsigned int i = 0; i < (size/2); ++i){
        h_B[i * 2] = h_signal[i].x;
        h_B[i * 2 + 1] = h_signal[i].y;
    }
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

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
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