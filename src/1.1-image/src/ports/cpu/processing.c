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
#include "image_kernels.h"

void proc_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i)
{
	/* [I]: Bias offset correction */
	T_START_VERBOSE(t->t_offset[frame_i]);
	f_offset(frame, &p->offsets);
	T_STOP_VERBOSE(t->t_offset[frame_i]);

	/* [II]: Bad pixel correction */
	T_START_VERBOSE(t->t_badpixel[frame_i]);
	f_mask_replace(frame, &p->bad_pixels);
	T_STOP_VERBOSE(t->t_badpixel[frame_i]);

	/* [III]: Radiation scrubbing */
	T_START_VERBOSE(t->t_scrub[frame_i]);
	// FIXME neighbour reference is not correct.
	f_scrub(frame, p->frames, p->num_frames, p->num_neigh);
	T_STOP_VERBOSE(t->t_scrub[frame_i]);

	/* [IV]: Gain correction */
	T_START_VERBOSE(t->t_gain[frame_i]);
	f_gain(frame, &p->gains);
	T_STOP_VERBOSE(t->t_gain[frame_i]);

	/* [V]: Spatial binning */
	T_START_VERBOSE(t->t_binning[frame_i]);
	f_2x2_bin(frame, &p->binned_frame);
	T_STOP_VERBOSE(t->t_binning[frame_i]);
	
	/* [VI]: Co-adding frames */
	T_START_VERBOSE(t->t_coadd[frame_i]);
	f_coadd(&p->image, &p->binned_frame);
	T_STOP_VERBOSE(t->t_coadd[frame_i]);
}

void proc_image_all(image_data_t *p, image_time_t *t)
{
	unsigned int frame_i;
	
	/* Loop through each frames and perform pre-processing. */
	T_START(t->t_test);
	for(frame_i=0; frame_i<p->num_frames; frame_i++)
	{
		T_START(t->t_frame[frame_i]);
		proc_image_frame(p, t, &p->frames[frame_i], frame_i);
		T_STOP(t->t_frame[frame_i]);

		// FIXME radiation correction frame handling to be implemented

	}
	T_STOP(t->t_test);
}


