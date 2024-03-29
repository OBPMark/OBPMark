/** 
 * \brief OBPMark "Image corrections and calibrations." processing task.
 * \file processing.c
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "processing.h"
#include "obpmark.h"
#include "obpmark_time.h"

void prepare_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i)
{
	/* [I]: Bias offset correction */ 
	f_offset(frame, &p->offsets);

	/* [II]: Bad pixel correction */
	f_mask_replace(frame, &p->bad_pixels);
}

void proc_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i)
{
	/* [III]: Radiation scrubbing */
	f_scrub(frame, p->frames, frame_i);

	/* [IV]: Gain correction */
	f_gain(frame, &p->gains);

	/* [V]: Spatial binning */
	f_2x2_bin(frame, &p->binned_frame);

	
	/* [VI]: Co-adding frames */
	f_coadd(&p->image_output, &p->binned_frame);

}


/* Kernel functions */

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
			//PIXEL(sum_frame,x,y) = PIXEL(add_frame,x,y);
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
			PIXEL(frame,x,y) = (uint16_t)((uint32_t)(PIXEL(frame,x,y)) * (uint32_t)(PIXEL(gains,x,y)) >> 16);
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
	//if (y_mid == 541 && x_mid == 140){printf("POS s x %d y %d value %u:%u\n",y_mid, x_mid, sum, n_sum);}
	mean = n_sum == 0 ? 0 : sum / n_sum;

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
				PIXEL(frame,x,y) = (uint16_t)f_neighbour_masked_sum(frame,mask, x,y);
			}
		}
	}
}

void f_scrub(
	frame16_t *frame,
	frame16_t *fs,
	unsigned int frame_i
	)
{
	unsigned int x,y;
	static unsigned int num_neighbour = 4;
	
	uint32_t sum;
	uint32_t mean;
	uint32_t thr;

	/* Generate scrubbing mask */
	for(x=0; x<frame->w; x++) 
	{
		for(y=0; y<frame->h; y++)
		{
			/* Sum temporal neighbours between -2 to + 2 without actual */ // FIXME unroll by 4
			sum = PIXEL(&fs[frame_i-2],x,y) + PIXEL(&fs[frame_i-1],x,y) +  PIXEL(&fs[frame_i+1],x,y) + PIXEL(&fs[frame_i+2],x,y);
			/* Calculate mean and threshold */
			mean = sum / (num_neighbour); 
			thr = 2*mean; 
			/* If above threshold, replace with mean of temporal neighbours */
			//if (y == 541 && x == 140){printf("%u POS s x %d y %d value %u:%u ---- %u %u %u %u\n",frame_i+2, y, x, thr, mean, PIXEL(&fs[frame_i-2],x,y), PIXEL(&fs[frame_i-1],x,y),PIXEL(&fs[frame_i+1],x,y),PIXEL(&fs[frame_i+2],x,y));}
			if(PIXEL(frame,x,y) > thr) {
				//printf("POS s x %d y %d value %u:%u\n", y, x, thr, mean);
				PIXEL(frame,x,y) = (uint16_t)mean; 
			}
		}
	}
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
			PIXEL(binned_frame,x2,y2)	= PIXEL(frame,x,y) +  PIXEL(frame,(x+1),y) + PIXEL(frame,x,(y+1)) + PIXEL(frame,(x+1),(y+1));
			++y2;
		}
		++x2;
	}
}

