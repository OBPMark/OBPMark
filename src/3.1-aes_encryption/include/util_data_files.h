/**
 * \file util_data_files.h 
 * \brief Benchmark #3.1  Input data generation.
 * \author Marc Sole (BSC)
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

#define DEFAULT_INPUT_FOLDER "../../data/input_data/3.1-aes_encryption"


int load_data_from_files(

	uint8_t *input_plaintext,
	uint8_t *key,
	uint8_t *iv,

	unsigned int data_length,
	unsigned int key_size,

	char *input_folder
	);

#endif // UTIL_DATA_FILES_H_
