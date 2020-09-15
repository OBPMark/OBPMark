/*
 * Utility functions for image processing benchmark -- dynamic memory implementation (not to be used in embedded benchmarks)
 * david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info: https://essr.esa.int/license/european-space-agency-community-license-v2-3-weak-copyleft
 */
#include "image_mem_util.h"

#include <stdlib.h> /* malloc(), calloc(), free() */

/* Allocation functions */

int frame8_alloc(
	frame8_t *frame,
	unsigned int f_width,
	unsigned int f_height
	)
{
	unsigned int x;

	frame->w = f_width;
	frame->h = f_height;

	frame->f = (uint8_t**)malloc(sizeof(uint8_t*)*frame->w);
	if(!frame->f) return 0;

	for(x=0; x<frame->w; x++) {
		frame->f[x] = (uint8_t*)calloc(frame->h, sizeof(uint8_t));
		if(!frame->f[x]) return 0;
	}
	
	return 1;
}

int frame16_alloc(
	frame16_t *frame,
	unsigned int f_width,
	unsigned int f_height
	)
{
	unsigned int x;

	frame->w = f_width;
	frame->h = f_height;

	frame->f = (uint16_t**)malloc(sizeof(uint16_t*)*frame->w);
	if(!frame->f) return 0;

	for(x=0; x<frame->w; x++) {
		frame->f[x] = (uint16_t*)calloc(frame->h, sizeof(uint16_t));
		if(!frame->f[x]) return 0;
	}
	
	return 1;
}

int frame32_alloc(
	frame32_t *frame,
	unsigned int f_width,
	unsigned int f_height
	)
{
	unsigned int x;

	frame->w = f_width;
	frame->h = f_height;

	frame->f = (uint32_t**)malloc(sizeof(uint32_t*)*frame->w);
	if(!frame->f) return 0;

	for(x=0; x<frame->w; x++) {
		frame->f[x] = (uint32_t*)calloc(frame->h, sizeof(uint32_t));
		if(!frame->f[x]) return 0;
	}
	
	return 1;
}

int framefp_alloc(
	framefp_t *frame,
	unsigned int f_width,
	unsigned int f_height
	)
{
	unsigned int x;

	frame->w = f_width;
	frame->h = f_height;

	frame->f = (float**)malloc(sizeof(float*)*frame->w);
	if(!frame->f) return 0;

	for(x=0; x<frame->w; x++) {
		frame->f[x] = (float*)calloc(frame->h, sizeof(float));
		if(!frame->f[x]) return 0;
	}
	
	return 1;
}

/* Free functions */

void frame8_free(
	frame8_t *frame
	)
{
	unsigned int x;

	for(x=0; x<frame->w; x++) {
		free(frame->f[x]);
	}
	free(frame->f);
}


void frame16_free(
	frame16_t *frame
	)
{
	unsigned int x;

	for(x=0; x<frame->w; x++) {
		free(frame->f[x]);
	}
	free(frame->f);
}


void frame32_free(
	frame32_t *frame
	)
{
	unsigned int x;

	for(x=0; x<frame->w; x++) {
		free(frame->f[x]);
	}
	free(frame->f);
}


void framefp_free(
	framefp_t *frame
	)
{
	unsigned int x;

	for(x=0; x<frame->w; x++) {
		free(frame->f[x]);
	}
	free(frame->f);
}

