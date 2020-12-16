#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#ifdef CUDA
#include <cuda_runtime.h>
#elif OPENCL
#include <CL/cl.hpp>
#elif OPENMP
#include <omp.h>
#elif HIP
#else
#endif
///////////////////////////////////////////////////////////////////////////////////////////////
// DEFINES
///////////////////////////////////////////////////////////////////////////////////////////////
// frames defines
#define MINIMUNWSIZE 1024
#define MINIMUNHSIZE 1024
#define MINIMUNFRAMES 1
// bits defines
#define MINIMUNBITSIZE 14
#define MAXIMUNBITSIZE 16
// Device defines
#ifdef CUDA
#define DEVICESELECTED 0
#define BLOCK_SIZE_PLANE 256
#define BLOCK_SIZE 16
#elif OPENCL
#define DEVICESELECTED 0
#define BLOCK_SIZE_PLANE 256
#define BLOCK_SIZE 16
#elif OPENMP
#define DEVICESELECTED 0
#elif HIP
#else
#define DEVICESELECTED 0
#endif

// Bad pixel defines with 900/1000 is a 0.1 % of bad pixel
#define BADPIXELTHRESHOLD 900
#define MAXNUMBERBADPIXEL 1000


///////////////////////////////////////////////////////////////////////////////////////////////
// OBJECT DECLARATION
///////////////////////////////////////////////////////////////////////////////////////////////
#ifdef CUDA
struct DeviceObject{
    int* image_input;
    int* processing_image;
    int* processing_image_error_free;
    int* spatial_reduction_image;
    int* image_output;
    int* correlation_table;
    int* gain_correlation_map;
    bool* bad_pixel_map;
    // timing GPU computation
    cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start;
	cudaEvent_t *stop;
};
#elif OPENCL
struct DeviceObject{
    cl::Context *context;
    cl::CommandQueue *queue;
	cl::Device default_device;
    cl::Program* program;
    // buffers
    cl::Buffer *image_input;
    cl::Buffer *processing_image;
    cl::Buffer *processing_image_error_free;
    cl::Buffer *spatial_reduction_image;
    cl::Buffer *image_output;
    cl::Buffer *correlation_table;
    cl::Buffer *gain_correlation_map;
    cl::Buffer *bad_pixel_map;
    // timing GPU computation
    cl::Event *memory_copy_device_a;
    cl::Event *memory_copy_device_b;
    cl::Event *memory_copy_device_c;
	cl::Event *memory_copy_host;
    struct timespec start;
	struct timespec end;
};
#elif OPENMP
struct DeviceObject
{
    int* image_input;
    int* processing_image;
    int* processing_image_error_free;
    int* spatial_reduction_image;
    int* image_output;
    int* correlation_table;
    int* gain_correlation_map;
    bool* bad_pixel_map;
    float elapsed_time;
};
#elif HIP
#else
struct DeviceObject
{
    int* image_input;
    int* processing_image;
    int* processing_image_error_free;
    int* spatial_reduction_image;
    int* image_output;
    int* correlation_table;
    int* gain_correlation_map;
    bool* bad_pixel_map;
    struct timespec start;
	struct timespec end;
};
#endif


///////////////////////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATION
///////////////////////////////////////////////////////////////////////////////////////////////
void init(DeviceObject *device_object, char* device_name);
void init(DeviceObject *device_object, int platform ,int device, char* device_name);
bool device_memory_init(DeviceObject *device_object, unsigned int size_image, unsigned int size_reduction_image);
void copy_memory_to_device(DeviceObject *device_object, int* correlation_table, int* gain_correlation_map, bool* bad_pixel_map , unsigned int size_image);
void copy_frame_to_device(DeviceObject *device_object, int* input_data, unsigned int size_image, unsigned int frame);
void process_full_frame_list (DeviceObject *device_object,int* input_data,unsigned int frames,unsigned int size_image,unsigned int w_size, unsigned int h_size);
void process_image(DeviceObject *device_object, unsigned int w_size, unsigned int h_size, unsigned int frame);
void copy_memory_to_host(DeviceObject *device_object, int* output_image, unsigned int size_image);
float get_elapsed_time(DeviceObject *device_object, bool csv_format);
void clean(DeviceObject *device_object);