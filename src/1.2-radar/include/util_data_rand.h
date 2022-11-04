/**
 * \file util_data_rand.h 
 * \brief Benchmark #1.2 random data generation.
 * \author Marc Sole Bonet (BSC)
 */

// FIXME 
// FIXME ********** THIS FUNCTION NEEDS TO BE REPLACED WITH THE EXACT SAME SCRIPT AS IN THE GENERATION OF THE VERIFICATION DATA ***************
// FIXME 

#ifndef UTIL_DATA_RAND_H_
#define UTIL_DATA_RAND_H_

#include "obpmark.h"
#include "benchmark.h"
#include "obpmark_image.h"
#include "device.h"

#define BENCHMARK_RAND_SEED	28012015

void benchmark_gen_rand_params(
        radar_params_t *params,
        unsigned int height,
        unsigned int width
        );

void benchmark_gen_rand_data(
        framefp_t *data,
        radar_params_t *params,
        unsigned int height, 
        unsigned int width
        );

#endif // UTIL_DATA_RAND_H_
