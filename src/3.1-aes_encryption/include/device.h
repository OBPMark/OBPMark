/**
 * \file device.h
 * \brief Benchmark #3.1 device definition.
 * \author Marc Sole (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "obpmark_time.h"
#include "benchmark.h"

/* Common Typedefs */
typedef enum {
	AES_KEY128 = 128,
	AES_KEY192 = 192,
	AES_KEY256 = 256
} AES_keysize_t;

typedef enum {
    AES_ECB,
    AES_CTR
} AES_mode_t;

/* Device Typedefs */
#ifdef CUDA
/* CUDA version */

#include <cuda_runtime.h>

typedef struct {
    uint8_t *value;
    AES_keysize_t size;
    uint32_t Nb;
    uint32_t Nk;
    uint32_t Nr;
} AES_key_t;

struct AES_values_t
{
	AES_key_t *key;
	uint8_t *plaintext;
	size_t data_length;
	uint8_t *expanded_key;
	uint8_t *cyphertext;
	uint8_t *iv;
	AES_mode_t mode;
	uint8_t *sbox;
	uint8_t *rcon;
};

struct AES_data_t{
    AES_values_t *host;
    AES_key_t *host_key;
    AES_values_t *dev;
};


typedef struct {
	cudaEvent_t *start;
	cudaEvent_t *stop;
    cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
} AES_time_t; 

#elif OPENCL
static const std::string type_def_kernel = "typedef short int uint16_t;\ntypedef unsigned char uint8_t;\ntypedef unsigned int uint32_t;\n";
/* OPENCL version */
#include <CL/cl.hpp>

typedef struct {
    cl::Buffer *value;
    AES_keysize_t size;
    uint32_t Nb;
    uint32_t Nk;
    uint32_t Nr;
} AES_key_t;

struct AES_data_t
{
    cl::Program *program;
    cl::Context *context;
	cl::Device default_device;
    cl::CommandQueue *queue;

	AES_key_t *key;
    cl::Buffer *plaintext;
	size_t data_length;
    cl::Buffer *expanded_key;
    cl::Buffer *cyphertext;
    cl::Buffer *iv;
	AES_mode_t mode;
    cl::Buffer *sbox;
    cl::Buffer *rcon;
};

typedef struct {
	cl::Event *start_test;
	cl::Event *stop_test;
	cl::Event *start_memory_copy_device;
	cl::Event *stop_memory_copy_device;
	cl::Event *memory_copy_host;

} AES_time_t; 

#elif HIP
/* HIP version */
#else
/* Sequential C version & OPENMP version */

typedef struct {
    uint8_t *value;
    AES_keysize_t size;
    uint32_t Nb;
    uint32_t Nk;
    uint32_t Nr;
} AES_key_t;

struct AES_data_t
{
	AES_key_t *key;
	uint8_t *plaintext;
	size_t data_length;
	uint8_t *expanded_key;
	uint8_t *cyphertext;
	uint8_t *iv;
	AES_mode_t mode;
	uint8_t *sbox;
	uint8_t *rcon;
};

typedef struct {
	time_t t_test;
} AES_time_t; 

#endif

/* Functions */
/**
 * \brief Basic init function to initialize  the target device.
 */
void init(
	AES_data_t *AES_data,
	AES_time_t *t,
	char *device_name
	);

/**
 * \brief Advance init function to initialize the target device. This is meant to be use when more that one device need to be selected of the same type.
 */
void init(
	AES_data_t *AES_data,
	AES_time_t *t,
	int platform,
	int device,
	char *device_name
	);

/**
 * \brief This function take cares of the initialization of the memory in the target device.
 */
bool device_memory_init(
	AES_data_t *AES_data,
    AES_mode_t enc_mode,
    unsigned int key_size,
    unsigned int data_length
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	AES_data_t *AES_data,
	AES_time_t *t,
	uint8_t *input_key,
	uint8_t *input_text,
	uint8_t *input_iv,
	uint8_t *input_sbox,
	uint8_t *input_rcon
	);

/**
 * \brief Main processing function that call the benchmark code.
 */
void process_benchmark(
	AES_data_t *AES_data,
	AES_time_t *t
	);

/**
 * \brief Function to copy the result from the device memory to the host memory.
 */
void copy_memory_to_host(
	AES_data_t *AES_data,
	AES_time_t *t,
	uint8_t *output
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
void get_elapsed_time(
	AES_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	);


/**
 * \brief Function to clean the memory in the device. 
 */
void clean(
	AES_data_t *AES_data,
	AES_time_t *t
	);

#endif // DEVICE_H_
