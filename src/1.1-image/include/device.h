/**
 * \file device.h
 * \brief Benchmark #1.1 device definition.
 * \author Ivan Rodrigues (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "benchmark.h"

/* Typedefs */

/* CUDA verion */
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

/* OpenCL version */
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

/* OpenMP version */
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

/* HIP version */
#elif HIP
// TODO HIP version 

#else
/* Sequential C version */
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

/* Functions */
// FIXME add brief function descriptions 

void init(
	DeviceObject *device_object,
	char *device_name
	);

void init(
	DeviceObject *device_object,
	int platform,
	int device,
	char *device_name
	);

bool device_memory_init(
	DeviceObject *device_object,
	unsigned int size_image,
	unsigned int size_reduction_image
	);

void copy_memory_to_device(
	DeviceObject *device_object,
	int *correlation_table,
	int *gain_correlation_map,
	bool *bad_pixel_map, 
	unsigned int size_image
	);

void copy_frame_to_device(
	DeviceObject *device_object,
	int *input_data, 
	unsigned int size_image, 
	unsigned int frame
	);

void process_full_frame_list(
	DeviceObject *device_object,
	int* input_data,
	unsigned int frames,
	unsigned int size_image,
	unsigned int w_size,
	unsigned int h_size
	);

void process_image(
	DeviceObject *device_object,
	unsigned int w_size,
	unsigned int h_size,
	unsigned int frame
	);

void copy_memory_to_host(
	DeviceObject *device_object,
	int *output_image,
	unsigned int size_image
	);

float get_elapsed_time(
	DeviceObject *device_object,
	bool csv_format
	);

void clean(DeviceObject *device_object);

#endif // DEVICE_H_
