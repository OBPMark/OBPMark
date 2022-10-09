/**
 * \file device.h
 * \brief Benchmark #1.1 device definition.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#ifndef DEVICE_H_
#define DEVICE_H_

#include "obpmark.h"
#include "util_prints.h"
#include "benchmark.h"
#include "obpmark_image.h"


/* Typedefs */
#ifdef CUDA
/* CUDA version */

struct image_data_t
{
	uint16_t *frames;
	unsigned int num_frames; 

	uint16_t *offsets;
	uint16_t *gains; 
	uint8_t *bad_pixels;

	uint8_t *scrub_mask;

	uint32_t *binned_frame; 

	uint32_t *image_output; 
};


typedef struct {
	cudaEvent_t *start_test;
	cudaEvent_t *stop_test;
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;

} image_time_t; 
#elif OPENCL
/* OPENCL version */
/* define the types to have the same as the cuda version */
#define uint32_t_cl unsigned int
#define uint16_t_cl unsigned short
#define uint8_t_cl unsigned char
static const std::string type_def_kernel = "#define uint32_t_cl unsigned int\n#define uint16_t_cl unsigned short\n#define uint8_t_cl unsigned char\n";
struct image_data_t
{
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device *default_device;
	cl::Program* program;

	cl::Buffer *frames;
	unsigned int num_frames; 

	cl::Buffer *offsets;
	cl::Buffer *gains; 
	cl::Buffer *bad_pixels;

	cl::Buffer *scrub_mask;

	cl::Buffer *binned_frame; 

	cl::Buffer *image_output; 
};

typedef struct {
	time_t t_test;
	time_t t_hots_device;
	cl::Event *t_device_host;

} image_time_t; 

#elif OPENMP
/* OPENMP version */
struct image_data_t
{
	frame16_t *frames;
	unsigned int num_frames; 

	frame16_t offsets;
	frame16_t gains; 
	frame8_t bad_pixels;

	frame8_t scrub_mask;

	frame32_t binned_frame; 

	frame32_t image_output; 
};

typedef struct {
	double t_test;


} image_time_t; 
#elif HIP
/* HIP version */
struct image_data_t
{
        uint16_t *frames;
        unsigned int num_frames;

        uint16_t *offsets;
        uint16_t *gains;
        uint8_t *bad_pixels;

        uint8_t *scrub_mask;

        uint32_t *binned_frame;

        uint32_t *image_output;
};


typedef struct {
        hipEvent_t *start_test;
        hipEvent_t *stop_test;
        hipEvent_t *start_memory_copy_device;
        hipEvent_t *stop_memory_copy_device;
        hipEvent_t *start_memory_copy_host;
        hipEvent_t *stop_memory_copy_host;

} image_time_t;



#else
/* Sequential C version */
struct image_data_t
{
	frame16_t *frames;
	unsigned int num_frames; 

	frame16_t offsets;
	frame16_t gains; 
	frame8_t bad_pixels;

	frame8_t scrub_mask;

	frame32_t binned_frame; 

	frame32_t image_output; 
};

typedef struct {
	time_t t_test;

} image_time_t; 
#endif
/* Functions */
// FIXME add brief function descriptions 

/**
 * \brief Basic init function to initialize  the target device.
 */
void init(
	image_data_t *image_data,
	image_time_t *t,
	char *device_name
	);

/**
 * \brief Advance init function to initialize the target device. This is meant to be use when more that one device need to be selected of the same type.
 */
void init(
	image_data_t *image_data,
	image_time_t *t,
	int platform,
	int device,
	char *device_name
	);

/**
 * \brief This function take cares of the initialization of the memory in the target device.
 */
bool device_memory_init(
	image_data_t *image_data,
	frame16_t* input_frames,
	frame16_t* offset_map, 
	frame8_t* bad_pixel_map, 
	frame16_t* gain_map,
	unsigned int w_size,
	unsigned int h_size
	);

/**
 * \brief This function is responsible for the copy of the memory from the host device to the target device
 */
void copy_memory_to_device(
	image_data_t *image_data,
	image_time_t *t,
	frame16_t *input_frames,
	frame16_t *correlation_table,
	frame16_t *gain_correlation_map,
	frame8_t *bad_pixel_map
	);

/**
 * \brief Main processing function that call the benchmark code.
 */
void process_benchmark(
	image_data_t *image_data,
	image_time_t *t,
	frame16_t *input_frames,
	unsigned int width,
    unsigned int height
	);

/**
 * \brief Function to copy the result from the device memory to the host memory.
 */
void copy_memory_to_host(
	image_data_t *image_data,
	image_time_t *t,
	frame32_t *output_image
	);

/**
 * \brief Function that summarize the execution time of the benchmark.
 */
void get_elapsed_time(
	image_data_t *image_data, 
	image_time_t *t, 
	print_info_data_t *benchmark_info,
	long int timestamp
	);


/**
 * \brief Function to clean the memory in the device. 
 */
void clean(
	image_data_t *image_data,
	image_time_t *t
	);

#endif // DEVICE_H_
