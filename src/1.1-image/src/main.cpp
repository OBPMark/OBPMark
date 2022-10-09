/**
 * \file main.c 
 * \brief Benchmark #1.1 benchmark main file.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
// FIXME copy top comment+license from old code 
// FIXME add license to all files 

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Optional utility headers */
#include "util_arg.h"
#include "util_data_rand.h"
#include "util_data_files.h"

void print_output_result(frame32_t *output_image)
{
	unsigned int h_position; 
	unsigned int w_position;

	/* Print output */
	for(h_position=0; h_position < output_image->h; h_position++)
	{
		
		for(w_position=0; w_position < output_image->w; w_position++)
		{
			//FIXME chaneg to the 1D and 2D version
			printf("%u, ", output_image->f[(h_position * (output_image->h) + w_position)]);
		}
		printf("\n");
	}
}

void init_benchmark(
	frame16_t *input_frames,
	frame32_t *output_image,

	frame16_t *offset_map,
	frame8_t *bad_pixel_map, 
	frame16_t *gain_map,
	
	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames,
	
	long int timestamp,
	print_info_data_t *benchmark_info
	)
{
	image_time_t *t = (image_time_t *)malloc(sizeof(image_time_t));
	image_data_t *image_data = (image_data_t *)malloc(sizeof(image_data_t));
	/* Init number of frames */
	image_data->num_frames = num_frames;
	char device[100] = "";
	char* output_file = (char*)"output.bin";

	/* Device object init */
	init(image_data, t, 0, DEVICESELECTED, device);

	print_device_info(benchmark_info, device);
	print_benchmark_info(benchmark_info);

	/* Initialize memory on the device and copy data */
	device_memory_init(image_data, input_frames, offset_map, bad_pixel_map, gain_map, w_size/2, h_size/2);
	copy_memory_to_device(image_data, t, input_frames, offset_map, gain_map, bad_pixel_map);

	/* Run the benchmark, by processing the full frame list */
	process_benchmark(image_data,t, input_frames, w_size, h_size);

	/* Copy data back from device */
	copy_memory_to_host(image_data, t, output_image);

	/* Get benchmark times */

	get_elapsed_time(image_data, t, benchmark_info, timestamp);
	if(benchmark_info->print_output)
	{
		print_output_result(output_image);
	}
	else 
	{
		// write the output image to a file call "output.bin"
		if (!benchmark_info->no_output_file)
		{
			write_frame32 (output_file, output_image, !benchmark_info->csv_mode && !benchmark_info->database_mode);
		}
		
	}

	/* Clean and free device object */
	clean(image_data, t);
}

int main(int argc, char **argv)
{
	int ret; 

	bool csv_mode = false;
	bool print_output = false;
	bool verbose_output = false;
	bool database_mode = false;
	bool random_data = false;
	bool no_output_file = false;

	int file_loading_output = 0;

	unsigned int w_size = 0;
	unsigned int h_size = 0; 
	unsigned int num_frames = 0; 
	unsigned int num_processing_frames = 0;

	unsigned int size_frame;
	unsigned int mem_size_frame;
	unsigned int mem_size_bad_map;
	unsigned int size_reduction_image;
	unsigned int mem_size_reduction_image;

	unsigned int frame_i;
	static unsigned int number_neighbours = 4;

	frame16_t *input_frames;
	frame32_t *output_image;

	frame16_t *offset_map;  
	frame8_t *bad_pixel_map; 
	frame16_t *gain_map;
	char input_folder[100] = "";

	/* Command line argument handling */
	ret = arguments_handler(argc, argv, &w_size, &h_size, &num_processing_frames, &csv_mode, &database_mode, &print_output, &verbose_output, &random_data, &no_output_file, input_folder);
	if(ret == ARG_ERROR) {
		exit(-1);
	}
	
	/* Assign the number of frame + the for extra neighbours */
	num_frames = num_processing_frames + number_neighbours;
	/* Assign frame sizes */
	size_frame		         = w_size * h_size;
	mem_size_frame		     = size_frame * sizeof(uint16_t);
	mem_size_bad_map	     = size_frame * sizeof(uint8_t);
	size_reduction_image	 = (w_size/2) * (h_size/2);
	mem_size_reduction_image = size_reduction_image* sizeof(uint32_t);

	/* Init print information data */
	print_info_data_t *benchmark_info = (print_info_data_t *)malloc(sizeof(print_info_data_t));
	benchmark_info -> w_size = w_size;
	benchmark_info -> h_size = h_size;
	benchmark_info -> num_frames = num_processing_frames;
	benchmark_info -> csv_mode = csv_mode;
	benchmark_info -> print_output = print_output;
	benchmark_info -> database_mode = database_mode;
	benchmark_info -> verbose_print = verbose_output;
	benchmark_info -> random_data = random_data;
	benchmark_info -> no_output_file = no_output_file;

	/* Allocate memory for frames */
	input_frames = (frame16_t*)malloc(sizeof(frame16_t)* num_frames);
	output_image = (frame32_t*)malloc(sizeof(frame32_t));
	/* Init internal frame data input*/
	// FIXME create 2D memory reservation
	for(frame_i=0; frame_i < num_frames; frame_i++)
	{
		input_frames[frame_i].f = (uint16_t*)malloc(mem_size_frame);
	}
	
	// FIXME create 2D memory reservation
	output_image->f = (uint32_t*)malloc(mem_size_reduction_image);
	output_image->h = w_size/2;
	output_image->w = h_size/2;


	/* Allocate memory for calibration and correction data */
	// FIXME create 2D memory reservation
	offset_map = (frame16_t*)malloc(sizeof(frame16_t));
	offset_map->f = (uint16_t*)malloc(mem_size_frame);
	// FIXME create 2D memory reservation
	bad_pixel_map = (frame8_t*)malloc(sizeof(frame8_t));
	bad_pixel_map->f = (uint8_t*)malloc(mem_size_bad_map);
	// FIXME create 2D memory reservation
	gain_map = (frame16_t*)malloc(sizeof(frame16_t));
	gain_map->f = (uint16_t*)malloc(mem_size_frame);

	if (random_data)
	{
		/* Generate random data */
		benchmark_gen_rand_data(
		input_frames, output_image,
		offset_map, bad_pixel_map, gain_map,
		w_size, h_size, num_frames
		);
	}
	else
	{
		/* Load data from files */
		
		file_loading_output = load_data_from_files(
		input_frames, output_image,
		offset_map, bad_pixel_map, gain_map,
		w_size, h_size, num_frames, input_folder
		);
		if (file_loading_output == FILE_LOADING_ERROR)
		{
			exit(-1);
		}
	}
	/* Init device and run test */
	init_benchmark(
		input_frames, output_image,
		offset_map, bad_pixel_map, gain_map,
		w_size, h_size, num_frames,get_timestamp(), benchmark_info
		);

	/* Free input data */
	for(frame_i=0; frame_i < num_frames; frame_i++)
	{
		free(input_frames[frame_i].f);
	}
	free(input_frames);
	free(output_image->f);
	free(offset_map->f);
	free(gain_map->f);
	free(bad_pixel_map->f);
	free(output_image);
	free(offset_map);
	free(gain_map);
	free(bad_pixel_map);
	free(benchmark_info);

	return 0;
}
