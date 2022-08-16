/**
 * \file device.h
 * \brief Benchmark #1.2 device definition.
 * \author Marc Sole Bonet (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "benchmark.h"
#include "obpmark_image.h"

typedef struct {
    float lambda;
    float PRF;
    float tau;
    float fs;
    float vr;
    float ro;
    float slope;
    uint32_t asize;     //total number of azimuth samples
    uint32_t avalid;    //number of meaningfull azimuth samples in a patch
    uint32_t apatch;    //total number of azimuth samples in a patch
    uint32_t rsize;     //total number of range samples = samples in a patch 
    uint32_t rvalid;    //valid range samples after range compress
    uint32_t npatch;    //number of patches in the image
}radar_params_t;

/* Typedefs */
#ifdef CUDA
#elif OPENCL
#elif OPENMP
/* OPENMP version */

struct radar_data_t
{
	framefp_t *range_data; //width: range, height: azimuth
	framefp_t *azimuth_data; //width: azimuth, height: range
	framefp_t ml_data;
	frame8_t output_image;
	float *rrf; //range reference function
	float *arf; // azimuth reference function
	uint32_t *offsets; //Offset table for RCMC
    float *aux; //Auxiliar data to compute Doppler centroid
	radar_params_t *params;
};

typedef struct {
    double t_test;
} radar_time_t; 

#elif HIP
#else
/* Sequential C version */
struct radar_data_t
{
	framefp_t *range_data; //width: range, height: azimuth
	framefp_t *azimuth_data; //width: azimuth, height: range
	framefp_t ml_data;
	frame8_t output_image;
	float *rrf; //range reference function
	float *arf; // azimuth reference function
	uint32_t *offsets; //Offset table for RCMC
    float *aux; //Auxiliar data to compute Doppler centroid
	radar_params_t *params;
};

typedef struct {
	time_t t_test;
} radar_time_t; 
#endif

/* Functions */
// FIXME add brief function descriptions 

/**
 * \brief Basic init function to initialize  the target device.
 */
void init(
	radar_data_t *radar_data,
	radar_time_t *t,
	char *device_name
	);

/**
 * \brief Advance init function to initialize the target device. This is meant to be use when more that one device need to be selected of the same type.
 */
void init(
	radar_data_t *radar_data,
	radar_time_t *t,
	int platform,
	int device,
	char *device_name
	);

/**
 * \brief This function take cares of the initialization of the memory in the target device.
 */
bool device_memory_init(
	radar_data_t *radar_data,
	radar_params_t *params,
    unsigned int out_height,
    unsigned int out_width
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	radar_data_t *radar_data,
	radar_time_t *t,
	framefp_t *input_data,
	radar_params_t *input_params
	);

/**
 * \brief Main processing function that call the benchmark code.
 */
void process_benchmark(
	radar_data_t *radar_data,
	radar_time_t *t
	);

/**
 * \brief Function to copy the result from the device memory to the host memory.
 */
void copy_memory_to_host(
	radar_data_t *radar_data,
	radar_time_t *t,
	frame8_t *output_radar
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
void get_elapsed_time(
	radar_data_t *radar_data, 
	radar_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	);


/**
 * \brief Function to clean the memory in the device. 
 */
void clean(
	radar_data_t *radar_data,
	radar_time_t *t
	);

#endif // DEVICE_H_
