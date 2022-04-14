/**
 * \file util_data_files.h 
 * \brief Benchmark #1.1  Input data generation.
 * \author Ivan Rodriguez (BSC)
 */

#ifndef UTIL_DATA_FILES_H_
#define UTIL_DATA_FILES_H_

#include "obpmark.h"

#include "benchmark.h"

#include "obpmark_image.h"
#include "image_mem_util.h"
#include "image_file_util.h"

#define FILE_LOADING_ERROR 3
#define FILE_LOADING_SUCCESS 0


int load_data_from_files(

	frame16_t *input_frames,
	frame32_t *output_frames, 
	
	frame16_t *offset_map,
	frame8_t *bad_pixel_map,
	frame16_t *gain_map,

	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames,

	char *input_folder
	);

#endif // UTIL_DATA_FILES_H_
