/**
 * \file datagen-1.1-image.c
 * \brief Data generation for Benchmark #1.1: Image calibrations and corrections
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#define OBPMARK_FRAME_DATA_2D // FIXME there is a bug in image_mem_util... it currently only works with 2D data... fix there. Also OBPMARK_FRAME_DATA_2D is defined in makefile
#include "obpmark_image.h"
#include "image_mem_util.h"
#include "image_file_util.h"

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1


int arguments_handler(int argc, char ** argv, unsigned int *frame_width, unsigned int *frame_height,unsigned int *num_frames, char *input_file);

void gen_offsets(frame16_t *offsets);
void gen_gains(frame16_t *gains);
void gen_bad_pixels(frame8_t *bad_pixels);

unsigned int add_offsets(frame16_t *frame, frame16_t *offsets); 
unsigned int add_gains(frame16_t *frame, frame16_t *gains);
void add_rad_hits(frame16_t *frame, unsigned int rad_num);
unsigned int add_bad_pixels(frame16_t *frame, frame8_t *bad_pixels, unsigned int x_offset, unsigned int y_offset);

void gen_base_frame(frame16_t *base_frame);
void copy_frame(frame16_t *frame, frame16_t *base_frame);

/* Calibration data generation */

void gen_offsets(frame16_t *offsets)
{
	unsigned int x,y;

	for(x=0; x<offsets->w; x++)
	{
		for(y=0; y<offsets->h; y++)
		{
			/* Using 2^10 as estimate of variance of pixel */
			PIXEL(offsets,x,y) = (uint32_t)(rand() % 128);

		}
	}
}

void gen_gains(frame16_t *gains)
{
	unsigned int x,y;

	// the 0.95 % value
	const static unsigned int base_value = 62257;
	// get the +/- 5 % 
	const static unsigned int max_gain_top_variance = 3276; 
	const static unsigned int max_gain_lower_variance = 3276;

	for(x=0; x<gains->w; x++)
	{
		for(y=0; y<gains->h; y++)
		{	
			//uint16_t gain_value = 65535;
			uint16_t gain_value = (rand() % ((base_value + max_gain_top_variance) - (base_value - max_gain_lower_variance) + 1)) + (base_value - max_gain_top_variance);
			PIXEL(gains,x,y) = gain_value;
		}
	}
}

void gen_bad_pixels(frame8_t *bad_pixels)
{
	unsigned int x,y;
	uint8_t val;
	unsigned int bad_pixel_count=0; 

	for(x=0; x<bad_pixels->w; x++)
	{
		for(y=0; y<bad_pixels->h; y++)
		{
			/* Using a worst case of 5% of pixels marked as bad pixels */
			val = (uint8_t)((rand() % 20 + 1) / 20) & 0x1;

			PIXEL(bad_pixels,x,y) = val;
			bad_pixel_count += PIXEL(bad_pixels,x,y);
		}
	}
}

/* Functions for adding detector effects to the data */

unsigned int add_offsets(frame16_t *frame, frame16_t *offsets)
{
	unsigned int x,y; 
	for(x=0; x<offsets->w; x++)
	{
		for(y=0; y<offsets->h; y++)
		{
			u_int32_t offset = PIXEL(offsets,x,y);
			u_int32_t value = PIXEL(frame,x,y);
			u_int32_t final = offset + value;
			// check that do not overflow
			if (final > 65535)
			{
				PIXEL(frame,x,y) = 65535;
				//printf("ERROR\n");
				
				
			}
			// check that do not underflow
			else if (final < 0){
				PIXEL(frame,x,y) = 0;
			}
			else
			{ 
				PIXEL(frame,x,y) = (u_int16_t)(final);
			}
			
		}
	}
}

unsigned int add_gains(frame16_t *frame, frame16_t *gains)
{
	unsigned int x,y; 
	for(x=0; x<gains->w; x++)
	{
		for(y=0; y<gains->h; y++)
		{
			//PIXEL(frame,x,y) = PIXEL(frame,x,y) / PIXEL(gains,x,y);
			u_int16_t  frame_value = PIXEL(frame,x,y);
			u_int32_t  pixel_value = (uint32_t)(frame_value) << 16;
			u_int32_t  gain_value = (uint32_t)(PIXEL(gains,x,y));
			uint32_t value = pixel_value/gain_value;
			//u_int16_t  value = ((uint32_t)(PIXEL(frame,x,y)) << 16 ) / (uint32_t)(PIXEL(gains,x,y));

			// check that do not overflow
			if (value > 65535)
			{
				PIXEL(frame,x,y) = 65535;
				//printf("ERROR\n");
				
				
			}
			// check that do not underflow
			else if (value < 0){
				PIXEL(frame,x,y) = 0;
			}
			else
			{ 
				PIXEL(frame,x,y) = (uint16_t)(value);
			}
		}
	}
}

void add_rad_hits(frame16_t *frame, unsigned int rad_num)
{
	
	const static unsigned int max_hit_top_variance = 10;
	const static unsigned int max_hit_lower_variance = 10;

	/* This will create the actual number of rad hits on the image, will use the rad_num value and will be a random value between */
	/* [(max_hit_lower_variance - rad_num)   ~(max_hit_top_variance + rad_num) ]*/ 
	unsigned int rad_num_adj = (rand() % ((rad_num + max_hit_top_variance) - (rad_num - max_hit_lower_variance) + 1)) + (rad_num - max_hit_lower_variance);

	unsigned int size_x = frame->w;
	unsigned int size_y = frame->h;
	unsigned int x_pos,y_pos;

	// TODO check if new methot is correct
	/*for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{*/
			/* Radiation hit chance */
			/*if((rand() % rad_prob) == 0) { */
				/* Add an arbitrarily large value with some variance */ 
				/*PIXEL(frame,x,y) += 32768 + (rand() % 4096);
				rad_count++;
			}
		}
	}
	return rad_count;*/

	for (unsigned int x = 0; x < rad_num_adj; ++x)
    {
        x_pos = (rand() % size_x );
        y_pos = (rand() % size_y );
        PIXEL(frame,x_pos,y_pos) += 32768 + (rand() % 4096);
    }
}

unsigned int add_bad_pixels(frame16_t *frame, frame8_t *bad_pixels, unsigned int x_offset, unsigned int y_offset)
{
	unsigned int x,y;
	unsigned int x2,y2;
	unsigned int bad_pixel_count = 0;

	for(x=x_offset; x<(frame->w - x_offset); x++)
	{
		x2 = x - x_offset;
		for(y=y_offset; y<(frame->h - y_offset); y++)
		{
			y2 = y - y_offset;
			if(PIXEL(bad_pixels,x2,y2) == 1)
			{
				/* Saturate pixel (note: not always the case) */
				PIXEL(frame,x,y) = 65535;
				bad_pixel_count++;
			}
		}
	}
	return bad_pixel_count;
}

/* Frame generation */

void gen_base_frame(
		frame16_t *base_frame
		)
{
	unsigned int x,y;

	for(x=0; x<base_frame->w; x++)
	{
		for(y=0; y<base_frame->h; y++)
		{
			/* Using uniformaly distributed values centered at 2^14 (16384, mid of dynamic range) with 2^13 variance (8192) */
			PIXEL(base_frame,x,y) = (uint32_t)(rand() % 2048) + 8192;
		}
	}


}

void copy_frame(
		frame16_t *frame,
		frame16_t *base_frame
	      )
{
	unsigned int x,y;
	const static unsigned int max_variation = 16;
	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			/* Use the value of the first frame */
			//PIXEL(frame,x,y) = PIXEL(base_frame,x,y);
			// generate value variance
			int32_t value = PIXEL(base_frame,x,y);
			int32_t variation = (rand() % ((max_variation) - (-max_variation) + 1)) + (-max_variation);
			// check that do not underflow
			if (value + variation > 65535)
			{
				PIXEL(frame,x,y) = 65535;
				
			}
			// check that do not overflow
			else if (value + variation < 0){
				PIXEL(frame,x,y) = 0;
			}
			else
			{ 
				PIXEL(frame,x,y) = (uint16_t)(value + variation);
			}
			
		}
	}
}


/* Generation of data set */

int benchmark1_1_write_files(
	frame16_t *offsets,
	frame16_t *gains,
	frame8_t *bad_pixels,
	frame16_t *fs, 
	unsigned int frame_width,
	unsigned int frame_height,
	unsigned int num_frames
	)
{
	unsigned int i;

	char offsets_name[50];
	char gains_name[50];
	char bad_pixels_name[50];

	char frame_name[50];

	sprintf(offsets_name,	 "out/offsets-%d-%d.bin", 	frame_width, frame_height);
	sprintf(gains_name, 	 "out/gains-%d-%d.bin", 	frame_width, frame_height);
	sprintf(bad_pixels_name, "out/bad_pixels-%d-%d.bin", 	frame_width, frame_height);

	printf("Writing calibration data to files...\n");
	if(!write_frame16(offsets_name, offsets)) {
		printf("error: failed to write offsets.\n");
		return 0;
	}

	if(!write_frame16(gains_name, gains)) {
		printf("error: failed to write gains.\n");
		return 0;
	}

	if(!write_frame8(bad_pixels_name, bad_pixels)) {
		printf("error: failed to write bad_pixels.\n");
		return 0;
	}

	/* Write frame data to files */
	printf("Writing frame data to files...\n");
	for(i=0; i<num_frames; i++)
	{
		sprintf(frame_name, "out/frame_%d-%d-%d.bin", i, frame_width, frame_height);
		if(!write_frame16(frame_name, &fs[i])) {
			printf("error: failed to write frame data: %d\n", i);
			return 0;
		}
	}
	return 1;
}

int benchmark1_1_data_gen(
	unsigned int frame_width,
	unsigned int frame_height,
	char *input_file,
	unsigned int num_frames
	)
{
	unsigned int i; 
	const static unsigned int num_frames_pre_process = 4;
	const static unsigned int num_of_pixels_affected_by_radiation = 50;
	unsigned int num_of_total_frames = num_frames_pre_process + num_frames;

	frame16_t offsets;
	frame16_t gains;
	frame8_t bad_pixels;
	frame16_t input_frame;

	frame16_t *fs;

	/* Allocate calibration data buffers */
	printf("Allocating calibration data buffers...\n");
	if(!frame16_alloc(&input_frame, frame_width, frame_height)) return 1; 
	if(!frame16_alloc(&offsets, frame_width, frame_height)) return 1; 
	if(!frame16_alloc(&gains, frame_width, frame_height)) return 1; 
	if(!frame8_alloc(&bad_pixels, frame_width, frame_height)) return 1;

	/* Alloc frame buffers */
	printf("Allocating frame buffers...\n");
	fs = (frame16_t*)malloc(sizeof(frame16_t)* num_of_total_frames);
	for(i=0; i<num_of_total_frames ; i++)
	{
		if(!frame16_alloc(&fs[i], frame_width, frame_height)) return 2; 
	}
	/* Read input data */
	// read the binary file
	printf("Reading input data...\n");

	if(!read_frame16(input_file, &input_frame)) return 3;

	/* Generate calibration data */
	printf("Generating calibration data...\n");
	gen_offsets(&offsets);
	gen_gains(&gains);
	gen_bad_pixels(&bad_pixels);

	/* Generate frame data */
	printf("Generating base frame data frames 0 and 1...\n");
	/* Generate frame 0 */
	copy_frame(&fs[0], &input_frame);
	/* Generate frame 1 */
	copy_frame(&fs[1], &input_frame);
	printf("Generating base frame data frames 2 and 3...\n");

	/* Generate frame 2 */
	copy_frame(&fs[2], &input_frame);
	add_gains(&fs[2], &gains);
	add_rad_hits(&fs[2], num_of_pixels_affected_by_radiation);
	/* Generate frame 3 */
	copy_frame(&fs[3], &input_frame);
	add_gains(&fs[3], &gains);
	add_rad_hits(&fs[3], num_of_pixels_affected_by_radiation);

	printf("Generating frame data...\n");
	for(i=num_frames_pre_process; i<num_of_total_frames; i++)
	{
		/* Generate frame */
		copy_frame(&fs[i], &input_frame);
	
		/* Add static detector effects (same for all frames) */
		add_offsets(&fs[i], &offsets);
		add_gains(&fs[i], &gains);
		add_bad_pixels(&fs[i], &bad_pixels, 0, 0); // FIXME check offsets

		/* Add radiation effects (different per frame) */
		add_rad_hits(&fs[i], num_of_pixels_affected_by_radiation); // FIXME check rad_num
	}


	/* Write calibration data to files */
	if(!benchmark1_1_write_files(&offsets, &gains, &bad_pixels, fs, frame_width, frame_height, num_of_total_frames)) {
		 /* Free buffers if error happen */
		frame16_free(&offsets);
		frame16_free(&gains);
		frame8_free(&bad_pixels);

		for(i=0; i<num_of_total_frames; i++)
		{
			frame16_free(&fs[i]);
		}
		free(fs);
		return 3;
	}

	/* Free buffers */
	frame16_free(&offsets);
	frame16_free(&gains);
	frame8_free(&bad_pixels);

	for(i=0; i<num_of_total_frames; i++)
	{
		frame16_free(&fs[i]);
	}
	free(fs);

	printf("Done.\n");

	return 0; 
}

/* Main */

int main(int argc, char *argv[])
{
	int ret;
	srand (28012015);
	/* Settings */
	unsigned int frame_width, frame_height, num_frames;
	char input_file[100];
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	int resolution = arguments_handler(argc,argv,&frame_width,&frame_height, &num_frames,input_file);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Data generation
	///////////////////////////////////////////////////////////////////////////////////////////////
	ret = benchmark1_1_data_gen(frame_width, frame_height, input_file, num_frames);
	if(ret != 0) return ret;

	return 0;
}

void print_usage(const char * appName)
{
	printf("Usage: %s -w Size -h Size -f number_of_frames -i input_file [-h]\n", appName);
	printf(" -w : set size of the width of the image \n");
	printf(" -h: set size of the height of the image \n");
	printf(" -f: number of frames generated \n");
	printf(" -i: base image \n");
}

int arguments_handler(int argc, char ** argv, unsigned int *frame_width, unsigned int *frame_height, unsigned int *frame_num, char *input_file){
	if (argc == 1){
		printf("-w, -h, -f, -i need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'w' : args +=1; *frame_width = atoi(argv[args]);break;
			case 'h' : args +=1; *frame_height = atoi(argv[args]);break;
			case 'f' : args +=1; *frame_num = atoi(argv[args]);break;
			case 'i' : args +=1; strcpy(input_file, argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if (*frame_width == 0 || *frame_height == 0 || *frame_num == 0){
		printf("\n-w and -h -f need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (strcmp(input_file, "") == 0){
		printf("\n-i need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	return OK_ARGUMENTS;			
}
