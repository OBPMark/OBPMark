/** 
 * \brief OBPMark "Image corrections and calibrations." processing task and image kernels.
 * \file processing.h
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "obpmark.h"
#include "device.h"
#include "obpmark_image.h" 
#include "obpmark_time.h"

/*typedef struct {
	frame16_t *frames;
	unsigned int num_frames; 
	unsigned int num_neigh; 

	frame16_t offsets;
	frame16_t gains; 
	frame8_t bad_pixels;

	frame8_t scrub_mask;

	frame32_t binned_frame; 
	frame32_t image;
} image_data_t;*/

/*typedef struct {
	time_t t_test;
	time_t *t_frame;

#if (OBPMARK_TIMING > 1)
	time_t *t_offset;
	time_t *t_badpixel;
	time_t *t_scrub;
	time_t *t_gain;
	time_t *t_binning;
	time_t *t_coadd;
#endif
} image_time_t; */


/** 
 * \brief Processing for a single frame.
 */
void proc_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i);

/**
 * \brief Entry function for Benchmark #1.1: Image corrections and calibrations.
 */
void proc_image_all(image_data_t *p, image_time_t *t); 

/* Kernel functions */

/** 
 * \brief Remove an offset frame from another frame. 
 */
void f_offset(
	frame16_t *frame,
	frame16_t *offsets
	);

/**
 * \brief Co-add pixels in a frame into a sum frame.
 */
void f_coadd(
	frame32_t *sum_frame,
	frame32_t *add_frame
	);

/**
 * \brief Multiply a frame by a gain frame, pixel by pixel.
 */
void f_gain(
	frame16_t *frame,
	frame16_t *gains	
	);

/**
 * \brief Replaced masked pixels with average of (good) neighbouring pixels, based on mask frame.
 * Used for both bad pixel correction and radiation scrubbing.
 */
void f_mask_replace(
	frame16_t *frame,
	frame8_t *mask
	);

/**
 * \brief Radiation scrubbing.
 */
void f_scrub(
        frame16_t *frame,
        frame16_t *fs,
        unsigned int frame_i
	);

/** 
 * \brief 2x2 binning. 
 */
void f_2x2_bin(
	frame16_t *frame,
	frame32_t *binned_frame
	);

#endif // PROCESSING_H_

