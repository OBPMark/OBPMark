/** 
 * \brief #OBPMark #1.1: Image pre-processsing. 
 * \file benchmark1_1.c
 * \author David Steenari
 */
#include <stdint.h>
#include "image_kernels.h"

/**
 * \brief Entry function for Benchmark #1.1:
 */
void benchmark_1_1_image_preproc(
		frame32_t *frames,
		unsigned int num_frames,
		
		frame32_t *offset_frame,
		uint8_t *bad_pixel_frame,
		float **gain_frame,

		frame32_t *binned_frame,
		frame32_t *image, 
	)
{
	unsigned int frame_i;
	unsigned int num_iter = 0; // FIXME radiation correction frames....
	
	frame32_t *frame;

	// FIXME add back timing info again (with additional macros for initializing timers etc.) 

	/*
	 * Loop through each frames and perform pre-processing.
	 */
	for(frame_i=0; frame_i<num_frames; frame_i++) // FIXME radiation correction frames...
	{
		frame = &frame[frame_i];

		/* [I]: Bias offset correction */
		f_offset(frame, offset_frame);

		/* [II]: Bad pixel correction */
		f_mask_replace(frame, bad_pixel_frame);

		/* [III]: Radiation scrubbing */
		f_scrub(frame, num_frames, num_neigh); // FIXME remove general num_ parameters

		/* [IV]: Gain correction */
		f_gain(frame, gain_frame);

		/* [V]: Spatial binning */
		f_2x2_bin(frame, binned_frame);
	
		/* [VI]: Co-adding frames */
		f_coadd(image, binned_frame);
	}
}
