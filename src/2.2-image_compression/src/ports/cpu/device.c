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
    
	// auxiliary structures

	int  **image_data = NULL;
	image_data = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		image_data[i] = (int *)calloc(w_size_padded, sizeof(int));
	}

	int  **transformed_image = NULL;
	transformed_image = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		transformed_image[i] = (int *)calloc(w_size_padded, sizeof(int));
	}

	unsigned int total_blocks =  (h_size_padded / BLOCKSIZEIMAGE )*(w_size_padded/ BLOCKSIZEIMAGE);
	int **block_string = NULL;
	block_string = (int **)calloc(total_blocks,sizeof(long *));
	for(unsigned int i = 0; i < total_blocks ; i++)
	{
		block_string[i] = (int *)calloc(BLOCKSIZEIMAGE * BLOCKSIZEIMAGE,sizeof(long));
	}

	// read the image data
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			image_data[i][j] = compression_data->input_image [i * h_size_padded + j];
		}
	}
	
	
	// start the 2D DWT operation 
	T_START(t->t_test);
	T_START(t->t_dwt);
	// pass to the 2D DWT
	dwt2D_compression_computation(compression_data, image_data, transformed_image, h_size_padded, w_size_padded);
	T_STOP(t->t_dwt);
	T_START(t->t_bpe);
	

	// build_block_string
	/*
	##########################################################################################################
	# This fuction takes the rearrange image and creates total_blocks
	# So a 8 by 8 data is store in block_string[0][0] to block_string[0][63]
	# each position in x of block_string contains 64 data of the image
	##########################################################################################################
	*/
	
	build_block_string(transformed_image, h_size_padded, w_size_padded,block_string);

	// compute the BPE
	compute_bpe(compression_data, block_string, total_blocks);
	
	// print the block_string
	/*for (unsigned int i = 0; i < total_blocks; ++i)
	{
		for (unsigned int j = 0; j < BLOCKSIZEIMAGE * BLOCKSIZEIMAGE; ++j)
		{
			printf("%d ",block_string[i][j]);
		}
		printf("\n");
	}*/
	
	/*
	##########################################################################################################
	*/
	// write the transformed image to a binary file
	// open the file
	/*FILE *fp = fopen("output.bin", "wb");
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
	}*/
	/*
	##########################################################################################################
	*/

	T_STOP(t->t_bpe);
	T_STOP(t->t_test);

	// clean image
	for(unsigned int i = 0; i < h_size_padded; i++){
			free(transformed_image[i]);
		}
	free(transformed_image);
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