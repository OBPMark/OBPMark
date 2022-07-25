/**
 * \file device.h
 * \brief Benchmark #2.2 device definition.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "benchmark.h"
#include "obpmark_time.h"


/* Typedefs */
#ifdef CUDA
/* CUDA version */
struct compression_image_data_t
{

};

typedef struct {
	cudaEvent_t *start_test;
	cudaEvent_t *stop_test;
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;

} compression_time_t; 
#elif OPENCL
/* OPENCL version */
/* define the types to have the same as the cuda version */
struct compression_image_data_t
{

};

typedef struct {
	time_t t_test;
	time_t t_hots_device;
	cl::Event *t_device_host;

} compression_time_t; 

#elif OPENMP
/* OPENMP version */

struct compression_image_data_t
{

};

typedef struct {
	double t_test;
} compression_time_t; 
#elif HIP
/* HIP version */
struct compression_image_data_t
{

};

typedef struct {
        hipEvent_t *start_test;
        hipEvent_t *stop_test;
        hipEvent_t *start_memory_copy_device;
        hipEvent_t *stop_memory_copy_device;
        hipEvent_t *start_memory_copy_host;
        hipEvent_t *stop_memory_copy_host;

} compression_time_t;



#else
/* Sequential C version */
struct compression_image_data_t
{

	// general data
	int *input_image;
	unsigned int w_size;
	unsigned int h_size;
	unsigned int segment_size;
	unsigned int bit_size;
	unsigned int pad_rows;
	unsigned int pad_columns;
	bool type_of_compression;


};

typedef struct {
	time_t t_test;
	time_t t_dwt;
	time_t t_bpe;

} compression_time_t; 
#endif
/* Functions */
// FIXME add brief function descriptions 

/**
 * \brief Basic init function to initialize  the target device.
 */
void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	);

/**
 * \brief Advance init function to initialize the target device. This is meant to be use when more that one device need to be selected of the same type.
 */
void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	int platform,
	int device,
	char *device_name
	);

/**
 * \brief This function take cares of the initialization of the memory in the target device.
 */
bool device_memory_init(
	compression_image_data_t *image_data
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	compression_image_data_t *image_data,
	compression_time_t *t
	);

/**
 * \brief Main processing function that call the benchmark code.
 */
void process_benchmark(
	compression_image_data_t *image_data,
	compression_time_t *t
	);

/**
 * \brief Function to copy the result from the device memory to the host memory.
 */
void copy_memory_to_host(
	compression_image_data_t *image_data,
	compression_time_t *t
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
void get_elapsed_time(
	compression_image_data_t *image_data, 
	compression_time_t *t,
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	);

/**
 * \brief Function to clean the memory in the device. 
 */
void clean(
	compression_image_data_t *image_data,
	compression_time_t *t
	);

#endif // DEVICE_H_
