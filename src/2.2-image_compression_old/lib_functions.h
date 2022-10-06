#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

#include "constants.h"

#include "easyBMP/EasyBMP.h"

#ifdef OPENCL
// OpenCL lib
#include <CL/cl.hpp>
#elif CUDA
// CUDA lib
#include <cuda_runtime.h>
#elif OPENMP
#include <omp.h>
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct DataObject{
   	#ifdef OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::Device default_device;
	cl::Event *evt_copy_mains;
	cl::Event *evt_copy_auxiliar_float_1;
	cl::Event *evt_copy_auxiliar_float_2;
	cl::Event *evt;
	cl::Event *evt_copy_back;
	cl::Buffer *input_image;
	cl::Buffer *input_image_float;
	cl::Buffer *transformed_float;
	cl::Buffer *high_filter;
	cl::Buffer *low_filter;
	cl::Buffer *transformed_image;
	cl::Buffer *output_image;
	cl::Buffer *final_transformed_image;
	cl::Buffer *coeff_image_regroup;
	cl::Buffer *block_string;
	struct timespec start_dwt;
	struct timespec end_dwt;
	struct timespec start_bpe;
	struct timespec end_bpe;

	cl::Program* program;


    #elif CUDA
	// CUDA PART

	int* input_image;
	float* input_image_float;
	float* transformed_float;
	float* high_filter;
	float* low_filter;
	int* transformed_image;
	int* output_image;
	int* final_transformed_image;
	int* coeff_image_regroup;
	long* block_string;

	
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start_dwt;
	cudaEvent_t *stop_dwt;
	cudaEvent_t *start_bpe;
	cudaEvent_t *stop_bpe;
	struct timespec start_bpe_cpu;
	struct timespec end_bpe_cpu;
	#else
	// CPU part
	float** procesed_image;
	struct timespec start_dwt;
	struct timespec end_dwt;
	struct timespec start_bpe;
	struct timespec end_bpe;
	#endif
	// comon variables
	unsigned int w_size;
	unsigned int h_size;
	unsigned int pad_rows;
	unsigned int pad_columns;
	bool encode;
	bool type;
	char* filename_input;
	char* filename_output;
	float elapsed_time_dwt;
	float elapsed_time_bpe;
};


void init(DataObject *device_object, char* device_name);
void init(DataObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(DataObject *device_object);
void encode_engine(DataObject *device_object, int* image_data_linear);
void copy_data_to_cpu(DataObject *device_object, int* image_data);
void get_elapsed_time(DataObject *device_object, bool csv_format);
void clean(DataObject *device_object);


#endif
