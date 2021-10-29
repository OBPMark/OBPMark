/**
 * \file main.c 
 * \brief Benchmark #1.1 benchmark main file.
 * \author Ivan Rodriguez (BSC)
 */
// FIXME copy top comment+license from old code 
// FIXME add license to all files 

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Optional utility headers */
#include "util_arg.h"
#include "util_data_rand.h"

void print_statistics(int *output_image, unsigned int w_size, unsigned int h_size)
{
	unsigned int h_position; 
	unsigned int w_position;

	/* Print statistics */
	for(h_position=0; h_position < h_size/2; h_position++)
	{
		
		for(w_position=0; w_position < w_size/2; w_position++)
		{
			printf("%hu, ", output_image[(h_position * (h_size/2) + w_position)]);
		}
		printf("\n");
	}
}

void init_device(
	int *input_frames,
	int *output_image,

	int *offset_map,
	bool *bad_pixel_map, 
	int *gain_map,
	
	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames,
	unsigned int size_frame,
	unsigned int size_frame_list, 
	unsigned int size_reduction_image,

	bool csv_mode, 
	bool print_output
	)
{
	DeviceObject *device_object = (DeviceObject *)malloc(sizeof(DeviceObject));
	char device[100] = "";

	/* Device object init */
	init(device_object, 0, DEVICESELECTED, device);

	if(!csv_mode){
		printf("Using device: %s\n", device);
	}

	/* Initialize memory on the device and copy data */
	device_memory_init(device_object, size_frame_list, size_reduction_image);
	copy_memory_to_device(device_object, offset_map, gain_map, bad_pixel_map, size_frame);

	/* Run the benchmark, by processing the full frame list */
	process_full_frame_list(device_object, input_frames, num_frames, size_frame, w_size, h_size);

	/* Copy data back from device */
	copy_memory_to_host(device_object, output_image, size_reduction_image);

	/* Get benchmark times */
	get_elapsed_time(device_object, csv_mode);
	if(print_output)
	{
		print_statistics(output_image, w_size, h_size);
	}

	/* Clean and free device object */
	clean(device_object);
	free(device_object);
}

int main(int argc, char **argv)
{
	int ret; 

	bool csv_mode = false;
	bool print_output = false;

	unsigned int w_size = 0;
	unsigned int h_size = 0; 
	unsigned int num_frames = 0; 
	unsigned int bitsize = 0;

	unsigned int size_frame_list;
	unsigned int size_frame;
	unsigned int mem_size_frame_list;
	unsigned int mem_size_frame;
	unsigned int mem_size_bad_map;
	unsigned int size_reduction_image;
	unsigned int mem_size_reduction_image;

	// FIXME these should use a stdint type. Also why are they 32-bit? Should be 16-bit. 
	int *input_frames;
	int *output_image;

	int *offset_map;  
	bool *bad_pixel_map; // FIXME is bool C99? 
	int *gain_map;

	/* Command line argument handling */
	ret = arguments_handler(argc, argv, &w_size, &h_size, &num_frames, &bitsize, &csv_mode, &print_output);
	if(ret == ARG_ERROR) {
		exit(-1);
	}

	/* Assign frame sizes */
	size_frame_list		= w_size * h_size * num_frames;
	size_frame		= w_size * h_size;
	mem_size_frame_list	= size_frame_list * sizeof(int);
	mem_size_frame		= size_frame * sizeof(int);
	mem_size_bad_map	= size_frame * sizeof(bool);
	size_reduction_image	= (w_size/2) * (h_size/2);
	mem_size_reduction_image = size_reduction_image* sizeof(int);

	/* Allocate memory for frames */
	input_frames = (int*)malloc(mem_size_frame_list);
	output_image = (int*)malloc(mem_size_reduction_image);

	/* Allocate memory for calibration and correction data */
	offset_map = (int*) malloc(mem_size_frame);
	bad_pixel_map = (bool*) malloc(mem_size_frame);
	gain_map = (int*) malloc(mem_size_frame);

	/* Generate random data */
	benchmark_gen_rand_data(
		bitsize,
		input_frames, output_image,
		offset_map, bad_pixel_map, gain_map,
		w_size, h_size, num_frames
		);

	/* Init device and run test */
	init_device(
		input_frames, output_image,
		offset_map, bad_pixel_map, gain_map,
		w_size, h_size, num_frames,
		size_frame, size_frame_list, size_reduction_image,
		csv_mode, print_output
		);

	/* Free input data */
	free(input_frames);
	free(output_image);
	free(offset_map);
	free(gain_map);
	free(bad_pixel_map);

	return 0;
}
