/** 
 * \brief #OBPMark #1.1: Image pre-processsing. 
 * \file benchmark1_1.c
 * \author David Steenari
 */
#include <stdint.h>
#include "image_kernels.h"
#include "../common/timing.h"

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
	
	T_INIT(t_test);

	T_INIT(t_offset);
	T_INIT(t_badpixel);
	T_INIT(t_scrub);
	T_INIT(t_gain);
	T_INIT(t_binning);
	T_INIT(t_coadd);

	// FIXME add back saving of timer data for each stage.

	/*
	 * Loop through each frames and perform pre-processing.
	 */
	T_START(t_test);
	for(frame_i=0; frame_i<num_frames; frame_i++) // FIXME radiation correction frames...
	{
		frame = &frame[frame_i];

		/* [I]: Bias offset correction */
		T_START(t_offset);
		f_offset(frame, offset_frame);
		T_STOP(t_offset);

		/* [II]: Bad pixel correction */
		T_START(t_badpixel);
		f_mask_replace(frame, bad_pixel_frame);
		T_STOP(t_badpixel);

		/* [III]: Radiation scrubbing */
		T_START(t_scrub);
		f_scrub(frame, num_frames, num_neigh); // FIXME remove general num_ parameters
		T_STOP(t_scrub);

		/* [IV]: Gain correction */
		T_START(t_gain);
		f_gain(frame, gain_frame);
		T_STOP(t_gain);

		/* [V]: Spatial binning */
		T_START(t_binning);
		f_2x2_bin(frame, binned_frame);
		T_STOP(t_binning);
	
		/* [VI]: Co-adding frames */
		T_START(t_coadd);
		f_coadd(image, binned_frame);
		T_STOP(t_coadd);
	}
	T_STOP(t_test);
}
