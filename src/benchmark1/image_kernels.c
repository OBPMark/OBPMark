/**
 * \brief OBPMark #1: Image Pre-Processing - Common processing kernel functions. 
 * \file image_kernels.c 
 * \author David Steenari
 */
// FIXME add licence info 
#include "image_kernels.h"

/* Defines */

/** \brief Meta definition for frame pixel access */
#define PIXEL(frame,x,y) ((frame)->f[x][y])

/* Functions */

void f_offset(
	frame32_t *frame,
	frame32_t *offset
	)
{
	unsigned int x,y; 

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			PIXEL(frame,x,y) -= PIXEL(offset,x,y);
		}
	}
}

void f_coadd(
	frame32_t *sum_frame,
	frame32_t *add_frame
	)
{
	unsigned int x,y; 

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			PIXEL(sum_frame,x,y) += PIXEL(add_frame,x,y);
		}
	}
}

void f_gain(
	frame32_t *frame,
	float **gain_frame	
	)
{
	unsigned int x,y;

	for(x=0; x<frame->w; x++)
	{
		for(y=0; y<frame->h; y++)
		{
			PIXEL(frame,x,y) = (uint32_t)(PIXEL(frame,x,y) * gain_frame[x][y]);
		}
	}
}

void f_bad_pixel_cor(
	frame32_t *frame, 
	uint8_t	**bad_pixel_frame
	)
{
	// FIXME Add from implementation.
}

void f_scrub(
	frame32_t *fs,
	unsigned int num_frames,
	unsigned int num_neigh
	)
{
	// FIXME Add from other implementation.
}

void f_2x2_bin(
	frame32_t frame,
	frame32_t binned_frame
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
			x2 += 1;
		}
		y2 += 1;
	}
}

