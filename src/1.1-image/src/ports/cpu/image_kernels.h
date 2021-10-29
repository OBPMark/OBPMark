/**
 * \brief OBPMark #1.1: Image Calibrations and Corrections -- kernel functions
 * \file image_kernels.h
 * \author David Steenari
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef OBPMARK_IMAGE_KERNELS_H_
#define OBPMARK_IMAGE_KERNELS_H_

#include "obpmark.h"
#include "obpmark_image.h"

/* Functions */

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
        unsigned int num_frames,
        unsigned int num_neighbours
	);

/** 
 * \brief 2x2 binning. 
 */
void f_2x2_bin(
	frame16_t *frame,
	frame32_t *binned_frame
	);

#endif // OBPMARK_IMAGE_KERNELS_H_

