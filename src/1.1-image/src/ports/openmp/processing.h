/** 
 * \brief OBPMark "Image corrections and calibrations." processing task and image kernels.
 * \file processing.h
 * \author ivan.rodriguez@bsc.es
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef PROCESSING_OPENMP_H_
#define PROCESSING_OPENMP_H_

#include "obpmark.h"
#include "device.h"
#include "obpmark_image.h" 
#include "obpmark_time.h"


/** 
 * \brief Processing for a single frame. This fuction finish the processing of a single frame starting by step III Radiation Scrubbing.
 */
void proc_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i);

/**
 * \brief Processing for a single frame. This function starts the processing of a single frame doing step I and II.
 */
void prepare_image_frame(image_data_t *p, image_time_t *t, frame16_t *frame, unsigned int frame_i);

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

#endif // PROCESSING_OPENMP_H_

