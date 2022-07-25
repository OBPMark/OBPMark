/**
 * \file device.c
 * \brief Benchmark #122 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */

#include "device.h"
#include "processing.h"


void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	)
{
    init(compression_data,t, 0,0, device_name);
}



void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	int platform,
	int device,
	char *device_name
	)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");

}


bool device_memory_init(
	compression_image_data_t *compression_data
	)
{
	// empty
return true;
}


void copy_memory_to_device(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	// empty
}


void process_benchmark(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{

	// first read the data and copy to the input image
    unsigned int size = sizeof(int) * compression_data->h_size * compression_data->w_size;
	// calculate the number of pading
	unsigned int h_size_padded = compression_data->h_size + compression_data->pad_rows;
	unsigned int w_size_padded = compression_data->w_size + compression_data->pad_columns;
	unsigned int pad_rows = compression_data->pad_rows;
	unsigned int pad_colums = compression_data->pad_columns;
    // read imput image

	int  **image_data = NULL;
	image_data = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		image_data[i] = (int *)calloc(w_size_padded, sizeof(int));
	}
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			image_data[i][j] = compression_data->input_image [i * h_size_padded + j];
		}
	}
	int  **transformed_image = NULL;
	transformed_image = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		transformed_image[i] = (int *)calloc(w_size_padded, sizeof(int));
	}
	
	// start the 2D DWT operation 
	T_START(t->t_test);
	T_START(t->t_dwt);
	// pass to the 2D DWT
	dwt2D_compression_computation(compression_data, image_data, transformed_image);
	T_STOP(t->t_dwt);
	T_START(t->t_bpe);
	// Step 1 transform the image 
	/*
	##########################################################################################################
	# This function take the image that has been processed for each of the levels of the DWT 2D and
	# re-arrange the data so each 8 by 8 block contains a family of de DC component been the DC component
	# in 0 0 of that block.
	##########################################################################################################
	*/

	coeff_regroup(transformed_image, h_size_padded, w_size_padded);
	// write the transformed image to a binary file
	// open the file
	FILE *fp = fopen("output.bin", "wb");
	if (fp == NULL)
	{
		printf("Error opening file!\n");
		return;
	}
	// write the data
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			fwrite(&transformed_image[i][j], sizeof(int), 1, fp);
		}
	}
	

	T_STOP(t->t_bpe);
	T_STOP(t->t_test);
}


void copy_memory_to_host(
	compression_image_data_t *image_data,
	compression_time_t *t
	)
{

}


void get_elapsed_time(
	compression_image_data_t *image_data, 
	compression_time_t *t,
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{
    
}



void clean(
	compression_image_data_t *image_data,
	compression_time_t *t
	)
{

}