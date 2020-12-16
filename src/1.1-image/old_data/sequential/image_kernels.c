/**
 * \brief OBPMark #1.1: Image Calibrations and Corrections -- kernel functions
 * \file image_kernels.c
 * \author David Steenari
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "image_kernels.h"

/* Functions */

void f_offset(
	frame16_t *frame,
	frame16_t *offsets
	)
{
	unsigned int x,y; 

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			PIXEL(frame,x,y) -= PIXEL(offsets,x,y);
		}
	}
}

void f_coadd(
	frame32_t *sum_frame,
	frame32_t *add_frame
	)
{
	unsigned int x,y; 

	for(x=0; x<sum_frame->w; x++)
	{
		for(y=0; y<sum_frame->h; y++)
		{
			PIXEL(sum_frame,x,y) += PIXEL(add_frame,x,y);
		}
	}
}

void f_gain(
	frame16_t *frame,
	frame16_t *gains	
	)
{
	unsigned int x,y;

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			// FIXME Q15 multiplication
			PIXEL(frame,x,y) = (uint16_t)(PIXEL(frame,x,y) * PIXEL(gains,x,y));
		}
	}
}

/**
 * \brief Help function for f_mask_replace(). Calculates mean of neighbours not marked in mask frame.
 */
uint32_t f_neighbour_masked_sum(
	frame16_t *frame,
	frame8_t *mask,
	int x_mid,
	int y_mid
	)
{
	int x,y;

	int x_start	= -1;
	int x_stop	= 1;
	int y_start	= -1;
	int y_stop	= 1;
	
	unsigned int n_sum=0;
	uint32_t sum=0;
	uint32_t mean; 

	/* Check if pixel is on edge of frame */
	if(x_mid == 0) {
		x_start = 0;
	}
	else if(x_mid == (frame->w-1)) {
		x_stop = 0;
	}

	if(y_mid == 0) {
		y_start = 0;
	}
	else if(y_mid == (frame->h-1)) {
		y_stop = 0;
	}

	/* Calculate unweighted sum of good pixels in 3x3 neighbourhood (can be smaller if on edge or corner). */
	for(x=x_start; x<(x_stop+1); x++)
	{
		for(y=y_start; y<(y_stop+1); y++)
		{
			/* Only include good pixels */
			if(PIXEL(mask,(x_mid+x),(y_mid+y)) == 0)
			{
				sum += PIXEL(frame,(x_mid+x),(y_mid+y));
				n_sum++;
			}
		}
	}

	/* Calculate mean of summed good pixels */
	mean = sum / n_sum;

	return mean;
}

void f_mask_replace(
	frame16_t *frame,
	frame8_t *mask
	)
{
	unsigned int x,y;

	/* Replace based on mask */
	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			if(PIXEL(mask,x,y) == 1)
			{
				/* Replace pixel value */
				PIXEL(frame,x,y) = f_neighbour_masked_sum(frame,mask, x,y);
			}
		}
	}
}

void f_scrub(
	frame16_t *frame,
	frame16_t *fs,
	frame8_t *scrub_mask,
	unsigned int num_frames,
	unsigned int num_neighbours
	)
{
	unsigned int x,y,i;
	
	uint32_t sum;
	float mean;
	uint32_t thr;

	/* Generate scrubbing mask */
	for(x=0; x<frame->w; x++) 
	{
		for(y=0; y<frame->h; y++)
		{
			/* Sum temporal neighbours */
			sum = 0;
			for(i=0;i<num_neighbours; i++)
			{
				sum += PIXEL(&fs[i],x,y);
				sum += PIXEL(&fs[i+num_neighbours+1],x,y);
			}
			/* Calculate mean and threshold */
			mean = sum / (2*num_neighbours); 
			thr = 2*mean; 

			/* Compare with threshold and mark result in scrubbing mask */
			if(PIXEL(frame,x,y) > thr) {
				PIXEL(scrub_mask,x,y) = 1; 
			}
			else {
				PIXEL(scrub_mask,x,y) = 0;
			}
		}
	}

	/* Scrub the frame using the generated mask */
	f_mask_replace(frame, scrub_mask);
}


void f_2x2_bin(
	frame16_t *frame,
	frame32_t *binned_frame
	)
{
	unsigned int x,y;
	unsigned int x2,y2;

	x2 = 0;
	for(x=0; x<frame->w; x+=2)
	{
		y2 = 0;
		for(y=0; y<frame->h; y+=2)
		{
			PIXEL(binned_frame,x2,y2)	= PIXEL(frame,x,y)	+ PIXEL(frame,(x+1),y)
							+ PIXEL(frame,x,(y+1))	+ PIXEL(frame,(x+1),(y+1));
			y2 += 1;
		}
		x2 += 1;
	}
}

