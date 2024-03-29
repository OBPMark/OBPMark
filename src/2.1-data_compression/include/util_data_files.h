/**
 * \file util_data_files.h 
 * \brief Benchmark #2.1  Input data generation.
 * \author Ivan Rodriguez (BSC)
 */

#ifndef UTIL_DATA_FILES_H_
#define UTIL_DATA_FILES_H_

#include "obpmark.h"
#include <stdio.h>

#include "benchmark.h"


#define FILE_LOADING_ERROR 3
#define FILE_LOADING_SUCCESS 0


int load_data_from_file(

	char * filename,
	unsigned int *data, 
	
	unsigned int j_blocksize,
    unsigned int r_samplesInterval,
    unsigned int steps
	);

int store_data_to_file(
    char * filename,
    unsigned char *data,
    unsigned int num_elements
    );

#endif // UTIL_DATA_FILES_H_