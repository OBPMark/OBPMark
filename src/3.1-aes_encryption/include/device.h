/**
 * \file device.h
 * \brief Benchmark #3.1 device definition.
 * \author Marc Sole (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "benchmark.h"
#include "obpmark_image.h"

/* Typedefs */
#ifdef CUDA
/* CUDA version */
#elif OPENCL
/* OPENCL version */
#elif OPENMP
/* OPENMP version */
#elif HIP
/* HIP version */
#else
/* Sequential C version */
/**
 * \brief Allowed AES key lengths.
 */
typedef enum {
	AES_KEY128 = 128,
	AES_KEY192 = 192,
	AES_KEY256 = 256
} AES_keysize_t;

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
	uint8_t *input_text;
	size_t data_length;
	uint8_t *expanded_key;
	uint8_t *encrypted_text;
	uint8_t *sbox;
	uint8_t *rcon;
};



typedef struct {
	time_t t_test;
	time_t *t_frame;
	// detailed timing
	time_t *t_offset;
	time_t *t_badpixel;
	time_t *t_scrub;
	time_t *t_gain;
	time_t *t_binning;
	time_t *t_coadd;

} AES_time_t; 

#endif
/* Functions */
// FIXME add brief function descriptions 

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
    unsigned int key_size,
    unsigned int data_size
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	AES_data_t *AES_data,
	uint8_t *input_key,
	uint8_t *input_text,
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
	uint8_t *output
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
//float get_elapsed_time(
//	image_data_t *image_data, 
//	image_time_t *t, 
//	bool csv_format
//	);


/**
 * \brief Function to clean the memory in the device. 
 */
void clean(
	AES_data_t *AES_data,
	AES_time_t *t
	);

#endif // DEVICE_H_
