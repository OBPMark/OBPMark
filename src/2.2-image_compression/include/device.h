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
#include "output_format_utils.h"

#include "math.h"

#define ABSOLUTE(a) ((a) >=0 ? (a): -(a))

#define NEGATIVE_SIGN 1
#define POSITIVE_SIGN 0
#define SIGN(var) ((var < 0) ? NEGATIVE_SIGN : POSITIVE_SIGN)

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
	unsigned int number_of_segments;
	bool type_of_compression;
	struct SegmentBitStream *segment_list;


};

struct header_data_t
{
	// part 1A
	bool start_img_flag = false; // 1 bit
	bool end_img_flag = false; // 1 bit
	unsigned char segment_count = 0; // 8 bits
	unsigned char bit_depth_dc = 0; // 5 bits
	unsigned char bit_depth_ac = 0; // 5 bits
	bool part_2_flag = false; // 1 bit
	bool part_3_flag = false; // 1 bit
	bool part_4_flag = false; // 1 bit
	// part 1B
	unsigned char pad_rows = 0; // 3 bits
	// with reserved bits the size of the header is 32 bits with 1A and 1B
	// with only 1A the size of the header is 24 bits

}; 

struct str_symbol_details_t
{
	unsigned char symbol_val;
	unsigned char symbol_len;
	unsigned char symbol_mapped_pattern;
	unsigned char sign;
	unsigned char type;

};

struct block_data_t
{
	unsigned long shifted_dc;
	unsigned long dc_reminder;
	unsigned long mapped_dc;
	unsigned long max_ac_bit_size;
	unsigned long mapped_ac;
	// plane history part
	unsigned char type_p;
	unsigned char tran_b;
	unsigned char tran_d;
	unsigned char tran_gi;
	unsigned char type_ci[3];
	unsigned char tran_hi[3];
	unsigned char type_hi[12];

	// srt symbol details
	str_symbol_details_t symbol_block[MAX_SYMBOLS_IN_BLOCK];
	// refinement symbol details
	unsigned char parent_ref_symbol;
	unsigned char parent_sym_len;
	unsigned short children_ref_symbol;
	unsigned char children_sym_len;
	unsigned short grand_children_ref_symbol[3];
	unsigned char grand_children_sym_len[3];

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
