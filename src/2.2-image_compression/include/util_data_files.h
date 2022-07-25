/**
 * \file util_data_files.h 
 * \brief Benchmark #2.2  Input data generation.
 * \author Ivan Rodriguez (BSC)
 */

#ifndef UTIL_DATA_FILES_H_
#define UTIL_DATA_FILES_H_

#include "obpmark.h"
#include <stdio.h>

#include "benchmark.h"
#include "device.h"

#define FILE_LOADING_ERROR 3
#define FILE_STORAGE_ERROR 2
#define FILE_LOADING_SUCCESS 0
#define FILE_STORAGE_SUCCESS 0


int load_data_from_file(
	char * filename,
	compression_image_data_t * ccdsd_data
	);

int store_data_to_file(
    char * filename,
    compression_image_data_t * ccdsd_data
    );

#endif // UTIL_DATA_FILES_H_