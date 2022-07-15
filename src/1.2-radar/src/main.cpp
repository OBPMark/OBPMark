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

void print_output_result(framefp_t *output_image)
{
	unsigned int h_position; 
	unsigned int w_position;

	/* Print output */
	for(h_position=0; h_position < output_image->h; h_position++)
	{
		
		for(w_position=0; w_position < output_image->w; w_position++)
		{
			//FIXME chaneg to the 1D and 2D version
			printf("%f, ",  output_image->f[(h_position * (output_image->w) + w_position)]);
		}
		printf("\n");
	}
}

void print_input_result(framefp_t *output_image)
{
	unsigned int h_position; 
	unsigned int w_position;
	printf("%d %d\n", output_image->h, output_image->w);

	/* Print output */
	for(h_position=0; h_position < output_image->h; h_position++)
	{
		
		for(w_position=0; w_position < output_image->w; w_position++)
		{
			//FIXME chaneg to the 1D and 2D version
			printf("%f, ", output_image->f[(h_position * (output_image->w) + w_position)]);
		}
		printf("\n");
	}
}

int write_output(char filename[], framefp_t *f)
{
	FILE *framefile;
	size_t bytes_written;
	size_t height = f->h;
	size_t width = f->w;
	size_t bytes_expected = height * sizeof(char);
	size_t bytes_total=0;
	unsigned int x, y;
//	uint8_t *vals = (uint8_t*) malloc(height*width);
//	for(x=0; x<height; x++)
//	    for(y=0; y<width; y++)
//	        vals[x*width+y] = (uint8_t) f->f[x*width+y];

	framefile = fopen(filename, "w");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}
	char aux[15];
    fprintf(framefile, "P2\n%d %d\n", width, height);
    for(x = 0; x<height; x++){
        for(y = 0; y<width; y++)
            fprintf(framefile, "%d ", (uint8_t) f->f[x*width+y]);
        fprintf(framefile, "\n");
    }

//	for(x=0; x<height; x++)
//	{
//		bytes_written = sizeof(float) * fwrite(&f->f[x*width], sizeof(float), width, framefile);
//		bytes_total += bytes_written;
//		if(bytes_written != bytes_expected) {
//			printf("error: writing file: %s, failed at row: %d, expected: %ld bytes, wrote: %ld bytes, total written: %ld bytes\n",
//					filename, x, bytes_expected, bytes_written, bytes_total);
//			return 0;
//		}
//	}

	fclose(framefile);
    printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*width));
	return 1;
}

void init_benchmark(
    framefp_t *input_data, 
    framefp_t *output_img,
    radar_params_t *params, 

	bool csv_mode, 
	bool print_output,
	bool database_mode,
	bool verbose_print,
	long int timestamp
	)
{
	radar_time_t *t = (radar_time_t *)malloc(sizeof(radar_time_t));
	radar_data_t *radar_data = (radar_data_t *)malloc(sizeof(radar_data_t));

	char device[100] = "";
	char* output_file = (char*)"output.bin";

	/* Device object init */
	init(radar_data, t, 0, DEVICESELECTED, device);

	if(!csv_mode){
		printf("Using device: %s\n", device);
	}

	/* Initialize memory on the device and copy data */
	device_memory_init(radar_data, params, input_data[0].h, input_data[0].w, output_img->h, output_img->w);
	copy_memory_to_device(radar_data, t, input_data, params);

	/* Run the benchmark, by processing the full frame list */
    process_benchmark(radar_data,t);

	/* Copy data back from device */
	copy_memory_to_host(radar_data, t, output_img);

	/* Get benchmark times */
	get_elapsed_time(radar_data, t, csv_mode, database_mode,verbose_print, timestamp);
	if(print_output)
		print_output_result(output_img);
	else 
		// write the output image to a file call "output.bin"
		write_output(output_file, output_img);

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

	int file_loading_output = 0;

	unsigned int in_height = 0;
	unsigned int in_width = 0;
	unsigned int ml_factor = 1;
	unsigned int out_height = 0;
	unsigned int out_width = 0;

	framefp_t *input_data;
    framefp_t *output_img;
	radar_params_t *params;


	char input_folder[100] = "";

	/* Command line argument handling */
	ret = arguments_handler(argc, argv, &in_height, &in_width, &ml_factor, &csv_mode, &database_mode, &print_output, &verbose_output, &random_data, input_folder);
	if(ret == ARG_ERROR) {
		exit(-1);
	}
	float ratio;
	if ( in_width > in_height ){
	    ratio = (float) in_width/ (float) in_height;
        out_width = ceil((float)in_width/(ratio * ml_factor));
        out_height = in_height / ml_factor;
	}
    else {
	    ratio = (float) in_height/ (float) in_width;
        out_width = in_width / ml_factor;
        out_height = ceil((float)in_height/(ratio * ml_factor));
    }
	params = (radar_params_t *) malloc(sizeof(radar_params_t));

	/* Read/Generate input parameters */
	if(random_data) benchmark_gen_rand_params(params,in_height, in_width);
    else if(load_params_from_file(params, in_height, in_width, input_folder) == FILE_LOADING_ERROR) exit(-1);

    unsigned int lines = pow(2,ceil(log2(params->asize+1)));
    unsigned int patchSize = lines * params->rsize * 2;
	input_data = (framefp_t*) malloc(sizeof(framefp_t)*params->npatch);
	for(int i = 0; i<params->npatch; i++){
	    input_data[i].f = (float*) malloc(sizeof(float) * patchSize);
	    input_data[i].h = lines;
	    input_data[i].w = params->rsize * 2;
	}

	output_img = (framefp_t*) malloc(sizeof(framefp_t));
    unsigned int out_size = out_height * out_width;

    output_img->f = (float *) malloc(sizeof(float)*out_size);
    output_img->w = out_width;
    output_img->h = out_height;
	
    /* Generate random data */
	if (random_data) benchmark_gen_rand_data(input_data, params, in_height, in_width);
	else if(load_data_from_files(input_data, params, in_height, in_width, input_folder) == FILE_LOADING_ERROR)exit(-1);

	/* Init device and run test */
	init_benchmark(input_data, output_img, params, csv_mode, print_output, database_mode, verbose_output, get_timestamp());

	for(int i = 0; i<params->npatch; i++)
        free(input_data[i].f);
	free(input_data);
	free(params);
	free(output_img->f);
	free(output_img);

	return 0;
}
