/**
 * \file util_data_rand.h 
 * \brief Benchmark #1.1 random data generation.
 * \author Ivan Rodriguez (BSC)
 */

// FIXME 
// FIXME ********** THIS FUNCTION NEEDS TO BE REPLACED WITH THE EXACT SAME SCRIPT AS IN THE GENERATION OF THE VERIFICATION DATA ***************
// FIXME 

#ifndef UTIL_DATA_RAND_H_
#define UTIL_DATA_RAND_H_

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

#define BENCHMARK_RAND_SEED	21121993

void benchmark_gen_rand_data(
	unsigned int bitsize,
	
	int *input_frames,
	int *output_image, 

	int *offset_map, 
	bool *bad_pixel_map,
	int *gain_map, 

	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames
	);

#endif // UTIL_DATA_RAND_H_
