/*
 * Utility functions for image processing benchmark -- dynamic memory implementation (not to be used in embedded benchmarks)
 * david.steenari@esa.int
 *
 * European Space Agency Community License V2.3 applies.
 * For more info: https://essr.esa.int/license/european-space-agency-community-license-v2-3-weak-copyleft
 */
#ifndef IMAGE_MEM_UTIL_H_
#define IMAGE_MEM_UTIL_H_

#include "obpmark_image.h"

/* Allocation functions */

int frame8_alloc(
	frame8_t *frame,
	unsigned int f_width,
	unsigned int f_height
	);

int frame16_alloc(
	frame16_t *frame,
	unsigned int f_width,
	unsigned int f_height
	);

int frame32_alloc(
	frame32_t *frame,
	unsigned int f_width,
	unsigned int f_height
	);

int framefp_alloc(
	framefp_t *frame,
	unsigned int f_width,
	unsigned int f_height
	);

/* Free functions */

void frame8_free(frame8_t *frame);

void frame16_free(frame16_t *frame);

void frame32_free(frame32_t *frame);

void framefp_free(framefp_t *frame);

#endif // IMAGE_MEM_UTIL_H_
