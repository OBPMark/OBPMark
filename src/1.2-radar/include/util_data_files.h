/**
 * \file util_data_files.h 
 * \brief Benchmark #1.2  Input data generation.
 * \author Marc Sole Bonet (BSC)
 */

#ifndef UTIL_DATA_FILES_H_
#define UTIL_DATA_FILES_H_

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

#include "obpmark_image.h"
#include "image_mem_util.h"
#include "image_file_util.h"

#define FILE_LOADING_ERROR 3
#define FILE_LOADING_SUCCESS 0


int load_data_from_files(
        framefp_t *input_data, 
        radar_params_t *params, 
        unsigned int height, 
        unsigned int width, 
        char *input_folder
	);
int load_params_from_file(
        radar_params_t *params, 
        unsigned int height,
        unsigned int width,
        char *input_folder
        );

#endif // UTIL_DATA_FILES_H_
