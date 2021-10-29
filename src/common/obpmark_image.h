/**
 * \brief OBPMark common definitions for image buffers used in benchmarks. 
 * \file obpmark_image.h
 * \author David Steenari (ESA)
 *
 * European Space Agency Community License V2.3 applies.
 * For more info: https://essr.esa.int/license/european-space-agency-community-license-v2-3-weak-copyleft
 */
#ifndef OBPMARK_IMAGE_H_
#define OBPMARK_IMAGE_H_

#include <stdint.h>

/* Defines */
#define frame_set(frame, buf, width, height) \
	frame.f = buf; \
	frame.w = width; \
	frame.h = height

/* Settting for 1D vs 2D buffers for storing and indexing frames */ 
#ifdef OBPMARK_FRAME_DATA_2D
	/* 2D buffers */
	#define PIXEL(frame,x,y) ((frame)->f[x][y]). 
#else
	/* 1D buffers */
	#define PIXEL(frame,x,y) ((frame)->f[y*(frame)->w + x])
#endif

/* Typedefs */
#ifdef OBPMARK_FRAME_DATA_2D
	/* 2D buffers */
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

#else 
	/* 1D buffers */
	typedef struct {
		uint8_t *f;   ///< Frame buffer
		unsigned int w; ///< Frame width
		unsigned int h; ///< Frame height
	} frame8_t;

	typedef struct {
		uint16_t *f;   ///< Frame buffer
		unsigned int w; ///< Frame width
		unsigned int h; ///< Frame height
	} frame16_t;

	typedef struct {
		uint32_t *f;   ///< Frame buffer
		unsigned int w; ///< Frame width
		unsigned int h; ///< Frame height
	} frame32_t;

	typedef struct {
		float *f;   ///< Frame buffer
		unsigned int w; ///< Frame width
		unsigned int h; ///< Frame height
	} framefp_t;
#endif


#endif // OBPMARK_IMAGE_H_
