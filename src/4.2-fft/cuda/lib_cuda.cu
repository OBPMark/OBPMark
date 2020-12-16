#include "../benchmark_library.h"

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void
binary_reverse_kernel(const bench_t *B, bench_t *Br, const int64_t size, const int group)
{
    
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int position = 0;
    if (id < size)
    {   
        position = (__brev(id) >> (32 - group)) * 2;
        Br[position] = B[id *2];
        Br[position + 1] = B[id *2 + 1];
    }
}

__global__ void
fft_kernel( bench_t *B, const int loop, const int inner_loop,const bench_t wr, const bench_t wi)
{   
    bench_t tempr, tempi;
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i;
    unsigned int j;
    // get I
    i = id *(loop * 2 * 2) + 1 + (inner_loop * 2); 
    j=i+(loop * 2 );

    tempr = wr*B[j-1] - wi*B[j];
    tempi = wr * B[j] + wi*B[j-1];
    
    B[j-1] = B[i-1] - tempr;
    B[j] = B[i] - tempi;
    B[i-1] += tempr;
    B[i] += tempi;
    
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


bool device_memory_init(GraficObject *device_object,  int64_t size_b_matrix){
    cudaError_t err = cudaSuccess;
    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    // Allocate the device reverse vector Br
    err = cudaMalloc((void **)&device_object->d_Br, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size){
    cudaError_t err = cudaSuccess;
    cudaEventRecord(*device_object->start_memory_copy_device);
    err = cudaMemcpy(device_object->d_B, h_B, sizeof(bench_t) * size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*device_object->stop_memory_copy_device);
    
}
void execute_kernel(GraficObject *device_object, int64_t size){
    dim3 dimBlock_reverse(BLOCK_SIZE);
    dim3 dimGrid_reverse(ceil(float(size)/dimBlock_reverse.x));
    dim3 dimBlock(0);
    dim3 dimGrid(0);

    bench_t wtemp, wpr, wpi, theta, wr, wi;

    cudaEventRecord(*device_object->start);
    // reorder kernel
    binary_reverse_kernel<<<dimGrid_reverse, dimBlock_reverse>>>(device_object->d_B, device_object->d_Br, size, (int64_t)log2(size));
    // Synchronize
    cudaDeviceSynchronize();
    // kernel call
    unsigned int theads = size /2 ;
    unsigned int loop = 1;
    //printf("size %d, dimBlock %d, dimGrid %d\n", size, dimBlock.x, dimGrid.x);
    //size = size << 1;
    while(loop < size ){
        // caluclate values 
        theta = -(M_PI/loop); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
        // calculate block size and thead size
        if (theads % BLOCK_SIZE != 0){
            // inferior part
            dimBlock.x = theads;
            dimGrid.x  = 1;
        }
        else{
            // top part
            dimBlock.x = BLOCK_SIZE;
            dimGrid.x  = (unsigned int)(theads/BLOCK_SIZE);
        }
        // launch kernel loop times
        for(unsigned int i = 0; i < loop; ++i){
            //kernel launch 
            fft_kernel<<<dimGrid, dimBlock>>>(device_object->d_Br, loop, i, wr, wi);
            // update WR, WI
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            
        }
        // update loop values
        loop = loop * 2;
        theads = theads / 2;
       
    }
   
    cudaEventRecord(*device_object->stop);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_B, device_object->d_Br, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
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

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
     err = cudaFree(device_object->d_Br);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector Br (error code %s)!\n", cudaGetErrorString(err));
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
