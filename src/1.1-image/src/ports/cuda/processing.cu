#include "benchmark_library.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// KERNELS
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void
image_offset_correlation_gain_correction(const int *image_input, const int *correlation_table, const int *gain_correlation_map,  int *processing_image, const int size_image)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < size_image)
    {
        //processing_image[x] =image_input[x];
        processing_image[x] =(image_input[x] - correlation_table[x]) * gain_correlation_map[x];
    }
}
__global__ void
bad_pixel_correlation(const int *processing_image, int * processing_image_error_free, const bool *bad_pixel_map, const unsigned int w_size ,const unsigned int h_size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w_size && y < h_size )
    {
        if (bad_pixel_map[y * h_size + x])
        {
            if (x == 0 && y == 0)
            {
                // TOP left
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x +1)] + processing_image[(y +1) * h_size +  (x +1) ] + processing_image[(y +1) * h_size + x  ])/3;
            }
            else if (x == 0 && y == h_size)
            {
                // Top right
                processing_image_error_free[y * h_size + x] = (processing_image[y * h_size +  (x -1)] + processing_image[(y -1) * h_size +  (x -1)] + processing_image[(y -1) * h_size + x ])/3;
            }
            else if(x == w_size && y == 0)
            {
                //Bottom left
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x] + processing_image[(y -1) * h_size +  (x + 1)] + processing_image[y * h_size +  (x +1)])/3;
            }
            else if (x == w_size && y == h_size)
            {
                // Bottom right
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  (x -1)] + processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1)])/3;
            }
            else if (y == 0)
            {
                // Top Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x -1) ] + processing_image[y * h_size +  (x +1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (x == 0)
            {
                //  Left Edge
                processing_image_error_free[y * h_size + x] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x +1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (x == w_size)
            {
                //  Right Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (y == h_size)
            {
                // Bottom Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1) ] + processing_image[y * h_size +  (x +1)])/3;
            }
            else
            {
                // Standart Case
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x -1)] + processing_image[y * h_size +  (x -1) ] + processing_image[(y +1) * h_size +  x  ] +  processing_image[(y +1) * h_size +  x  ])/4;
            }
        }
        else{
            processing_image_error_free[y * h_size + x ] = processing_image[y * h_size + x];
        }

    }
}
__global__ void
spatial_binning_temporal_binning(const int *processing_image, int *output_image, const unsigned int w_size_half ,const unsigned int h_size_half)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w_size_half && y < h_size_half )
    {
        output_image[y * h_size_half + x ] += processing_image[ (2*y)* (h_size_half*2) + (2 *x) ] + processing_image[(2*y)* (h_size_half*2) + (2 *(x+1))  ] + processing_image[(2*(y+1))* (h_size_half*2) + (2 *x) ] + processing_image[(2*(y+1))* (h_size_half*2) + (2 *(x+1)) ];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////
// CUDA FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////
void init(DeviceObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(DeviceObject *device_object, int platform ,int device, char* device_name){
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
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

bool device_memory_init(DeviceObject *device_object, unsigned int size_image, unsigned int size_reduction_image){
    // Allocate input
     cudaError_t err = cudaSuccess;
     err = cudaMalloc((void **)&device_object->image_input, size_image * sizeof(int));
 
     if (err != cudaSuccess)
     {
         return false;
     }
     // Allocate procesing image 
     err = cudaMalloc((void **)&device_object->processing_image, size_image * sizeof(int));
     if (err != cudaSuccess)
     {
         return false;
     }
     err = cudaMalloc((void **)&device_object->processing_image_error_free, size_image * sizeof(int));
     if (err != cudaSuccess)
     {
         return false;
     }
     err = cudaMalloc((void **)&device_object->image_output, size_reduction_image * sizeof(int));
     if (err != cudaSuccess)
     {
         return false;
     }
     err = cudaMalloc((void **)&device_object->correlation_table, size_image * sizeof(int));
     if (err != cudaSuccess)
     {
         return false;
     }
     err = cudaMalloc((void **)&device_object->gain_correlation_map, size_image * sizeof(int));
     if (err != cudaSuccess)
     {
         return false;
     }
     err = cudaMalloc((void **)&device_object->bad_pixel_map, size_image * sizeof(bool));
     if (err != cudaSuccess)
     {
         return false;
     }
     return true;
}
void copy_memory_to_device(DeviceObject *device_object, int* correlation_table, int* gain_correlation_map, bool* bad_pixel_map , unsigned int size_image){
    cudaEventRecord(*device_object->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(device_object->correlation_table , correlation_table , sizeof(int) * size_image, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector correlation_table from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->gain_correlation_map , gain_correlation_map , sizeof(int) * size_image, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector gain_correlation_map from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->bad_pixel_map , bad_pixel_map , sizeof(bool) * size_image, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector bad_pixel_map from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*device_object->stop_memory_copy_device);
}
void copy_frame_to_device(DeviceObject *device_object, int* input_data, unsigned int size_image, unsigned int frame){
    // record time
    cudaError_t err = cudaMemcpy(device_object->image_input , input_data + (frame * size_image), sizeof(int) * size_image, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    // record time
}

void process_full_frame_list (DeviceObject *device_object,int* input_frames,unsigned int frames, unsigned int size_frame,unsigned int w_size, unsigned int h_size){
    cudaEventRecord(*device_object->start);
    for (unsigned int frame = 0; frame < frames; ++frame )
    {
        // copy image
        copy_frame_to_device(device_object, input_frames, size_frame, frame);
        // process image
        process_image(device_object, w_size, h_size, frame);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(*device_object->stop);
}

void process_image(DeviceObject *device_object, unsigned int w_size, unsigned int h_size, unsigned int frame){
    unsigned int size_image = w_size * h_size;
    
    // image offset correlation Gain correction
    dim3 dimBlock, dimGrid;
    dimBlock = dim3(BLOCK_SIZE_PLANE);
    dimGrid = dim3(ceil(float(size_image)/dimBlock.x));
    image_offset_correlation_gain_correction<<<dimGrid, dimBlock>>>(device_object->image_input, device_object->correlation_table,device_object->gain_correlation_map,device_object->processing_image,size_image);
    // Bad pixel correction
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid = dim3(ceil(float(w_size)/dimBlock.x), ceil(float(h_size)/dimBlock.y));
    bad_pixel_correlation<<<dimGrid, dimBlock>>>(device_object->processing_image, device_object->processing_image_error_free,device_object->bad_pixel_map,w_size,h_size);
    // spatial Binning Temporal Binning
    dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dimGrid = dim3(ceil(float((w_size)/2)/dimBlock.x), ceil(float((h_size)/2)/dimBlock.y));
    spatial_binning_temporal_binning<<<dimGrid, dimBlock>>>(device_object->processing_image_error_free,device_object->image_output,w_size/2,h_size/2);

}


void copy_memory_to_host(DeviceObject *device_object, int* output_image, unsigned int size_image){
    cudaEventRecord(*device_object->start_memory_copy_host);
    //cudaMemcpy(output_image, device_object->processing_image_error_free, size_image * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(output_image, device_object->image_output, size_image * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
}

float get_elapsed_time(DeviceObject *device_object, bool csv_format){
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



void clean(DeviceObject *device_object){
    cudaError_t err = cudaSuccess;

    err = cudaFree(device_object->image_input);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector image_input (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaFree(device_object->processing_image);
    

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector processing_image (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->processing_image_error_free);
    

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector processing_image (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->image_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector image_output (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->correlation_table);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector correlation_table (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->gain_correlation_map);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector gain_correlation_map (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->bad_pixel_map);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector bad_pixel_map (error code %s)!\n", cudaGetErrorString(err));
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
