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
#include "device.h"

#define BENCHMARK_RAND_SEED	21121993

void benchmark_gen_rand_data(
	unsigned int bitsize, 

	int *input_frames,
	int *output_frames, 
	
	int *offset_map,
	bool *bad_pixel_map,
	int *gain_map,

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

	// selection of the 14 or 16 bits
	if(bitsize == MAXIMUNBITSIZE) {
		// UP TO 16 bits
		randnumber = 65535;
	}
	else if(bitsize == MINIMUNBITSIZE) {
		// UP TO 14 bits
		randnumber = 16383;
	}
	else {
		// DEFAULT 16 bits
		randnumber = 65535;
	}

	/* Input frame */
	for(frame_position=0; frame_position < num_frames; frame_position++)
	{
		for(w_position=0; w_position < w_size; w_position++)
		{
			for(h_position=0; h_position < h_size; h_position++)
			{
				// Fill with random data
				input_frames[((h_position * h_size) + w_position) + (frame_position * h_size * w_size)] = (int)rand() % randnumber;

			}
		}
	}

	// offset correlation init 
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			offset_map[((h_position * h_size) + w_position)] = (int)rand() % randnumber;
		}
	}

	// gain correction table
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			gain_map[((h_position * h_size) + w_position)] = (int)rand() % randnumber;
		}
	}

	// bad pixel correction
	for(w_position=0; w_position < w_size; w_position++)
	{
		for(h_position=0; h_position < h_size; h_position++)
		{
			// Fill with random data
			bad_pixel_map[((h_position * h_size) + w_position)] = (rand() % MAXNUMBERBADPIXEL) < BADPIXELTHRESHOLD;
		}
	}

}
