#include <stdio.h>
#include <stdlib.h>

#define true 1
#define false 0
#define bool unsigned char

#ifdef OPENCL
// OpenCL lib

#elif CUDA
// CUDA lib
#include <cuda_runtime.h>
#elif OPENMP
#include <omp.h>
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct DataObject
{
   	#ifdef OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::Device default_device;
	cl::Program* program;

	cl::Buffer *input_data;
	cl::Buffer *output_data;
	cl::Buffer *input_data_post_process;
	cl::Buffer *missing_value;
	cl::Buffer *missing_value_inverse;
	cl::Buffer *zero_block_list;
	cl::Buffer *zero_block_list_inverse;

	cl::Buffer *compresion_identifier;
	cl::Buffer  *size_block;
	cl::Buffer *data_in_blocks;

	cl::Buffer *compresion_identifier_best;
	cl::Buffer  *size_block_best;
	cl::Buffer *data_in_blocks_best;
	cl::Buffer *data_in_blocks_best_post_process;

	cl::Event *memory_copy_device;
	cl::Event *memory_copy_host_1;
	cl::Event *memory_copy_host_2;
	cl::Event *memory_copy_host_3;

	unsigned char *compresion_identifier_best_cpu;
	unsigned int  *size_block_best_cpu;
	unsigned long int *data_in_blocks_best_cpu;


	struct timespec start_app;
    struct timespec end_app;
    #elif CUDA
	// CUDA PART
    unsigned long int *input_data;
	unsigned long int *output_data;
	unsigned long int *input_data_post_process;
	int *missing_value;
	int *missing_value_inverse;
	int *zero_block_list;
	int *zero_block_list_inverse;

	unsigned char *compresion_identifier;
	unsigned int  *size_block;
	unsigned long int *data_in_blocks;

	unsigned char *compresion_identifier_best;
	unsigned int  *size_block_best;
	unsigned int *bit_block_best;
	unsigned long int *data_in_blocks_best;
	unsigned long int *data_in_blocks_best_post_process;

	unsigned char *compresion_identifier_best_cpu;
	unsigned int  *size_block_best_cpu;
	unsigned long int *data_in_blocks_best_cpu;
	//int *zero_block_list_cpu;
	//int *zero_block_list_inverse_cpu;

	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;

	struct timespec start_app;
    struct timespec end_app;

	#elif OPENMP
	float elapsed_time;
	#else
	
	// CPU exclusive
	struct timespec start_app;
    struct timespec end_app;
	#endif
	// comon variables
	unsigned long int *InputDataBlock;
	unsigned long int *OutputDataBlock;
	unsigned int TotalSamples;
	unsigned int TotalSamplesStep;
};


void init(struct DataObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(struct DataObject *device_object);
void execute_benchmark(struct DataObject *device_object);
void get_elapsed_time(struct DataObject *device_object, bool csv_format);
void clean(struct DataObject *device_object);


#endif
