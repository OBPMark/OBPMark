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

	uint8_t *input_plaintext,
	uint8_t *key,
	uint8_t *iv,

	unsigned int data_length,
	unsigned int key_size
	);

#endif // UTIL_DATA_FILES_H_
