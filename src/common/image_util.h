/*
 * Image processing benchmarks utility
 * david.steenari@esa.int
 *
 * European Space Agency Community License V2.3 applies.
 * For more info: https://essr.esa.int/license/european-space-agency-community-license-v2-3-weak-copyleft
 */
#ifndef IMAGE_UTIL_H_
#define IMAGE_UTIL_H_

#include <stdint.h>

/* Defines */
#define PIXEL(frame,x,y) ((frame)->f[x][y]) // Replace this macro if using 1D buffers for storing frames. 

#define frame_set(frame, buf, width, height) \
	frame.f = buf; \
	frame.w = width; \
	frame.h = height

/* Typedefs */

typedef struct {
	uint8_t **f;   ///< Frame buffer
	unsigned int w; ///< Frame width
	unsigned int h; ///< Frame height
} frame8_t;

typedef struct {
	uint16_t **f;   ///< Frame buffer
	unsigned int w; ///< Frame width
	unsigned int h; ///< Frame height
} frame16_t;

typedef struct {
	uint32_t **f;   ///< Frame buffer
	unsigned int w; ///< Frame width
	unsigned int h; ///< Frame height
} frame32_t;

typedef struct {
	float **f;   ///< Frame buffer
	unsigned int w; ///< Frame width
	unsigned int h; ///< Frame height
} framefp_t;

#endif // IMAGE_UTIL_H_
