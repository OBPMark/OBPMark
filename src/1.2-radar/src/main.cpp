/**
 * \file main.c 
 * \brief Benchmark #1.2 benchmark main file.
 * \author Marc Sole Bonet (BSC)
 */

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Optional utility headers */
#include "util_arg.h"
#include "util_data_rand.h"
#include "util_data_files.h"

void print_output_result(frame8_t *output_image)
{
	unsigned int h_position; 
	unsigned int w_position;

	/* Print output */
	for(h_position=0; h_position < output_image->h; h_position++)
	{
		
		for(w_position=0; w_position < output_image->w; w_position++)
		{
			//FIXME chaneg to the 1D and 2D version
			printf("%d, ",  output_image->f[(h_position * (output_image->w) + w_position)]);
		}
		printf("\n");
	}
}

void init_benchmark(
    framefp_t *input_data, 
    frame8_t *output_img,
    radar_params_t *params, 

	long int timestamp,
	print_info_data_t *benchmark_info
	)
{
	/* Alloc data containers */
	radar_time_t *t = (radar_time_t *)malloc(sizeof(radar_time_t));
	radar_data_t *radar_data = (radar_data_t *)malloc(sizeof(radar_data_t));

	char device[100] = "";
    // generate output filename that is output_<r_size>_<a_size>_IMPLEMENTATION_NAME_FILE.bin
    char output_filename[100] = "output_";
    char r_size_str[10];
    char a_size_str[10];
    // generate output filename
    sprintf(r_size_str, "%d", params->rsize);
    sprintf(a_size_str, "%d", params->asize);
    strcat(output_filename, r_size_str);
    strcat(output_filename, "_");
    strcat(output_filename, a_size_str);
    strcat(output_filename, "_");
    strcat(output_filename, IMPLEMENTATION_NAME_FILE);
    strcat(output_filename, ".bin");
    char *output_file = (char*) output_filename;
    benchmark_info->output_file = output_file;

	/* Device object init */
	init(radar_data, t, 0, DEVICESELECTED, device);

	print_device_info(benchmark_info, device);
	print_benchmark_info(benchmark_info);

	/* Initialize memory on the device and copy data */
	if(!device_memory_init(radar_data, params, output_img->h, output_img->w)){
	    printf("Error initializing memory\n");
	    return;
	}
	copy_memory_to_device(radar_data, t, input_data, params);


	/* Run the benchmark, by processing the full frame list */
    process_benchmark(radar_data,t);

	/* Copy data back from device */
	copy_memory_to_host(radar_data, t, output_img);

	/* Get benchmark times */
	get_elapsed_time(radar_data, t, benchmark_info, timestamp);
	
	if(benchmark_info->print_output)
    {
        print_output_result(output_img);
    }
	else 
    {
		// write the output image to a file call "output.bin"
        if (!benchmark_info->no_output_file)
        {
            int result = write_frame8(output_file, output_img, 0);
			if (result == 1 && !benchmark_info->csv_mode && !benchmark_info->database_mode)
			{
				printf("Done. Outputs written to %s\n", output_file);
			}
		}
    }

	/* Clean and free device object */
    clean(radar_data, t);
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
	bool extended_csv_mode = false;

	int file_loading_output = 0;

	unsigned int in_height = 0;
	unsigned int in_width = 0;
	unsigned int ml_factor = 1;
	unsigned int out_height = 0;
	unsigned int out_width = 0;

	framefp_t *input_data;
    frame8_t *output_img;
	radar_params_t *params;


	char input_folder[100] = "";

	/* Command line argument handling */
	ret = arguments_handler(argc, argv, &in_height, &in_width, &ml_factor, &csv_mode, &database_mode, &print_output, &verbose_output, &random_data, &no_output_file, &extended_csv_mode, input_folder);
	if(ret == ARG_ERROR) {
		exit(-1);
	}
    // If no folder is set, use default
    if(strcmp(input_folder,"") == 0)
        sprintf(input_folder, "%s/%d_%d", DEFAULT_INPUT_FOLDER, in_width, in_height);
    // If random data, no folder is used
	if(random_data) input_folder[0]=0;

	/* Create data to hold input parameters */
	params = (radar_params_t *) malloc(sizeof(radar_params_t));
	/* Read/Generate input parameters */
	if(random_data) benchmark_gen_rand_params(params,in_height, in_width);
    else if(load_params_from_file(params, in_height, in_width, input_folder) == FILE_LOADING_ERROR) exit(-1);

	/* Find output image size */
    out_width = params->rvalid / ml_factor;
	float azi_factor =(float) params->asize/(float)out_width; //height;
	out_height = floor(params->avalid/azi_factor)*params->npatch;

	print_info_data_t *benchmark_info = (print_info_data_t *)malloc(sizeof(print_info_data_t));
	benchmark_info -> w_size = params->rsize;
	benchmark_info -> h_size = params->asize;
	benchmark_info -> num_patch = params->npatch;
	benchmark_info -> ml_factor = ml_factor;
	benchmark_info -> out_width = out_width;
	benchmark_info -> out_height = out_height;
	benchmark_info -> csv_mode = csv_mode;
	benchmark_info -> print_output = print_output;
	benchmark_info -> database_mode = database_mode;
	benchmark_info -> verbose_print = verbose_output;
	benchmark_info -> random_data = random_data;
	benchmark_info -> no_output_file = no_output_file;
	benchmark_info -> input_folder = input_folder;
	benchmark_info -> extended_csv_mode = extended_csv_mode;

    /* Allocate input data */
    unsigned int patchSize = params->apatch * params->rsize * 2;
	input_data = (framefp_t*) malloc(sizeof(framefp_t)*params->npatch);
	for(int i = 0; i<params->npatch; i++){
	    input_data[i].f = (float*) malloc(sizeof(float) * patchSize);
	    input_data[i].h = params->apatch;
	    input_data[i].w = params->rsize * 2;
	}

	/* Allocate output data */
	output_img = (frame8_t*) malloc(sizeof(frame8_t));
    unsigned int out_size = out_height * out_width;

    output_img->f = (uint8_t *) malloc(sizeof(uint8_t)*out_size);
    output_img->w = out_width;
    output_img->h = out_height;

    /* Generate random data */
	if (random_data) benchmark_gen_rand_data(input_data, params, in_height, in_width);
	else if(load_data_from_files(input_data, params, in_height, in_width, input_folder) == FILE_LOADING_ERROR)exit(-1);

	/* Init device and run test */
	init_benchmark(input_data, output_img, params, get_timestamp(), benchmark_info);

	for(int i = 0; i<params->npatch; i++) free(input_data[i].f);
	free(input_data);
	free(params);
	free(output_img->f);
	free(output_img);
	free(benchmark_info);

	return 0;
}
