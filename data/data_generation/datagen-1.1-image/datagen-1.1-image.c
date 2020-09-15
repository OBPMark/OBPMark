/**
 * \file datagen-1.1-image.c
 * \brief Data generation for Benchmark #1.1: Image calibrations and corrections
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#include <stdio.h>
#include <stdlib.h>

#include "image_util.h"
#include "image_mem_util.h"
#include "image_file_util.h"

void gen_offsets(frame16_t *offsets);
void gen_gains(frame16_t *gains);
void gen_bad_pixels(frame8_t *bad_pixels);

void gen_base_frame(frame16_t *base_frame);
void gen_frame(frame16_t *frame, frame16_t *base_frame);

/* Calibration data generation */

void gen_offsets(frame16_t *offsets)
{
	unsigned int x,y;

	for(x=0; x<offsets->w; x++)
	{
		for(y=0; y<offsets->h; y++)
		{
			/* Using 2^10 as estimate of variance of pixel */
			PIXEL(offsets,x,y)	= (uint32_t)(rand() % 128);

		}
	}
}

void gen_gains(frame16_t *gains)
{
	unsigned int x,y;

	for(x=0; x<gains->w; x++)
	{
		for(y=0; y<gains->h; y++)
		{
			// FIXME gain function
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

/* Radiation and bad pixels */

unsigned int add_rad_hits(frame16_t* frame, unsigned int rad_num)
{
	unsigned int x,y;
	unsigned int rad_prob = (frame->w*frame->h)*rad_num;
	unsigned int rad_count = 0;

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			/* Radiation hit chance */
			if((rand() % rad_prob) == 0) {
				/* Add an arbitrarily large value with some variance */ 
				PIXEL(frame,x,y) += 32768 + (rand() % 4096);
				rad_count++;
			}
		}
	}
	return rad_count;
}

unsigned int add_bad_pixels(frame16_t* frame, frame8_t *bad_pixels, unsigned int x_offset, unsigned int y_offset)
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
			if(bad_pixels->f[x2][y2] == 1)
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

void gen_frame(
		frame16_t *frame,
		frame16_t *base_frame
	      )
{
	// FIXME use gain, offsets and bad pixels
	unsigned int x,y;

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			/* Use the value of the first frame and add some noise */
			PIXEL(frame,x,y) = PIXEL(base_frame,x,y) + ((rand() % 64) - (rand() % 64));
		}
	}
}

/* Generation of data set */

int benchmark1_1_data_gen(
	unsigned int frame_width,
	unsigned int frame_height,
	unsigned int num_frames
	)
{
	int i; 

	frame16_t offsets;
	frame16_t gains;
	frame8_t bad_pixels; 

	frame16_t *fs;

	/* Allocate calibration data buffers */
	printf("Allocating calibration data buffers...\n");
	if(!frame16_alloc(&offsets, frame_width, frame_height)) return 1; 
	if(!frame16_alloc(&gains, frame_width, frame_height)) return 1; 
	if(!frame8_alloc(&bad_pixels, frame_width, frame_height)) return 1;

	/* Alloc frame buffers */
	printf("Allocating frame buffers...\n");
	fs = (frame16_t*)malloc(sizeof(frame16_t)*num_frames);
	for(i=0; i<num_frames; i++)
	{
		if(!frame16_alloc(&fs[i], frame_width, frame_height)) return 2; 
	}

	/* Generate calibration data */
	printf("Generating calibration data...\n");
	gen_offsets(&offsets);
	gen_gains(&gains);
	gen_bad_pixels(&bad_pixels);

	/* Generate frame data */
	printf("Generating frame data...\n");
	// FIXME add offsets, gains and bad pixels
	gen_base_frame(&fs[0]);
	for(i=1; i<num_frames; i++)
	{
		gen_frame(&fs[i], &fs[0]);
	}


	/* Write calibration data to files */
	printf("Writing calibration data to files...\n");
	// FIXME pre-defined file_names -- should include frame width and height
	if(!write_frame16("out/image_offsets.bin", &offsets)) {
		printf("error: failed to write offsets.\n");
		// FIXME cleanup
		return 3;
	}

	if(!write_frame16("out/image_gains.bin", &gains)) {
		printf("error: failed to write gains.\n");
		// FIXME cleanup
		return 3;
	}

	if(!write_frame8("out/image_bad_pixels.bin", &bad_pixels)) {
		printf("error: failed to write bad_pixels.\n");
		// FIXME cleanup
		return 3;
	}

	/* Write frame data to files */
	printf("Writing frame data to files...\n");
	for(i=0; i<num_frames; i++)
	{
		// FIXME add frame width, height and frame number to output file name
		if(!write_frame16("out/frame_data.bin", &fs[i])) {
			printf("error: failed to write frame data: %d\n", i);
			// FIXME cleanup
			return 4;
		}
	}

	/* Free buffers */
	frame16_free(&offsets);
	frame16_free(&gains);
	frame8_free(&bad_pixels);

	for(i=0; i<num_frames; i++)
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
	/* Settings */
	int num_frames;
	int frame_width, frame_height; 

	// FIXME input parameters handling
	num_frames = 8;
	frame_width = 2048;
	frame_height = 2048;	

	return benchmark1_1_data_gen(frame_width, frame_height, num_frames);	
}
