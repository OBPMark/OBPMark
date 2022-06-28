/**
 * \file device.h
 * \brief Benchmark #1.1 device definition.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */

#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "obpmark_time.h"
#include "output_format_utils.h"
#include "benchmark.h"
#include "math.h"

/* Typedefs */
#ifdef CUDA
/* CUDA version */
struct compression_data_t
{

	unsigned int *input_data;
	unsigned int *output_data;
	unsigned int *input_data_post_process;
	int *missing_value;
	int *missing_value_inverse;
	int *zero_block_list;
	int *zero_block_list_inverse;

	unsigned char *compresion_identifier;
	unsigned char *compresion_identifier_internal;
	unsigned int *halved_samples;
	unsigned int  *size_block;
	unsigned int *data_in_blocks;

	unsigned char *compresion_identifier_best;
	unsigned char *compresion_identifier_internal_best;
	unsigned int  *size_block_best;
	unsigned int *bit_block_best;
	unsigned int *data_in_blocks_best;

	unsigned char *compresion_identifier_best_cpu;
	unsigned char *compresion_identifier_best_internal_cpu;
	unsigned int  *size_block_best_cpu;
	unsigned int *data_in_blocks_best_cpu;

	// general part

	unsigned int *InputDataBlock;
	unsigned int *OutputPreprocessedValue;
	struct OutputBitStream *OutputDataBlock;
	unsigned int n_bits;
	unsigned int j_blocksize;
	unsigned int r_samplesInterval;
	unsigned int steps;
	bool preprocessor_active;
	unsigned int TotalSamples;
	unsigned int TotalSamplesStep;
};
typedef struct {
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	time_t t_test;
}compression_time_t;

#elif OPENCL
/* OPENCL version */
/* define the types to have the same as the cuda version */
#define uint32_t_cl unsigned int
#define uint16_t_cl unsigned short
#define uint8_t_cl unsigned char
static const std::string type_def_kernel = "#define uint32_t_cl unsigned int\nt#define uint16_t_cl unsigned short\n#define uint8_t_cl unsigned char\n";
struct compression_data_t
{
};
typedef struct {
	
}compression_time_t;

#elif OPENMP
/* OPENMP version */
struct compression_data_t
{
};
typedef struct {
}compression_time_t;

#elif HIP
struct compression_data_t
{
};
typedef struct {
}compression_time_t;

#else
/* Sequential C version */
struct compression_data_t
{
	unsigned int *InputDataBlock;
	unsigned int *OutputPreprocessedValue;
	struct OutputBitStream *OutputDataBlock;
	unsigned int n_bits;
	unsigned int j_blocksize;
	unsigned int r_samplesInterval;
	unsigned int steps;
	bool preprocessor_active;
	unsigned int TotalSamples;
	unsigned int TotalSamplesStep;
};
typedef struct {
	time_t t_test;
}compression_time_t;
#endif
/* Functions */

/**
 * \brief Basic init function to initialize  the target device.
 */
void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	);

/**
 * \brief Advance init function to initialize the target device. This is meant to be use when more that one device need to be selected of the same type.
 */
void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	int platform,
	int device,
	char *device_name
	);

/**
 * \brief This function take cares of the initialization of the memory in the target device.
 */
bool device_memory_init(
	compression_data_t *compression_data
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	compression_data_t *compression_data,
	compression_time_t *t
	);

/**
 * \brief Main processing function that call the benchmark code.
 */
void process_benchmark(
	compression_data_t *compression_data,
	compression_time_t *t
	);

/**
 * \brief Function to copy the result from the device memory to the host memory.
 */
void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
void get_elapsed_time(
	compression_data_t *compression_data, 
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
	compression_data_t *compression_data,
	compression_time_t *t
	);


#endif // DEVICE_H_