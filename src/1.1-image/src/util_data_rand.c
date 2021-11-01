/**
 * \file util_data_rand.c 
 * \brief Benchmark #1.1 random data generation.
 * \author Ivan Rodriguez (BSC)
 */

// FIXME 
// FIXME ********** THIS FUNCTION NEEDS TO BE REPLACED WITH THE EXACT SAME SCRIPT AS IN THE GENERATION OF THE VERIFICATION DATA ***************
// FIXME 

#include "obpmark.h"

#include "benchmark.h"
#
#include "device.h"

#define BENCHMARK_RAND_SEED	21121993

void benchmark_gen_rand_data(

	frame16_t *input_frames,
	frame16_t *output_frames, 
	
	frame16_t *offset_map,
	frame8_t *bad_pixel_map,
	frame16_t *gain_map,

	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames
	)
{
	unsigned int frame_position; 
	unsigned int w_position;
	unsigned int h_position;

	int randnumber = 0;

	/* Initialize srad seed */
	srand(BENCHMARK_RAND_SEED);
	// DEFAULT 16 bits
	randnumber = 65535;
	
	/* Input frame */
	input_frames->w = w_size;
	input_frames->h = h_size;
	for(frame_position=0; frame_position < num_frames; frame_position++)
	{
		for(w_position=0; w_position < w_size; w_position++)
		{
			for(h_position=0; h_position < h_size; h_position++)
			{
				// Fill with random data
				// FIXME think how to do this with pixels or allow for 2D matrix
				PIXEL(&input_frames[frame_position], w_position,h_position) = (uint16_t)rand() % randnumber;

			}
		}
	}

	// offset correlation init 
	offset_map->w = w_size;
	offset_map->h = h_size;
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			PIXEL(offset_map, w_position,h_position) =  (uint16_t)rand() % randnumber;
		}
	}

	// gain correction table
	gain_map->w = w_size;
	gain_map->h = h_size;
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			PIXEL(gain_map, w_position,h_position) =  (uint16_t)rand() % randnumber;
		}
	}

	// bad pixel correction
	bad_pixel_map->w = w_size;
	bad_pixel_map->h = h_size;
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			PIXEL(bad_pixel_map, w_position,h_position) =  (uint8_t)rand() % randnumber;
		}
	}

}
