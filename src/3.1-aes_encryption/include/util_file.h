/**
 * \file util_file.c
 * \brief File reader for benchmark #3.1
 * \author Marc Solé Bonet (BSC)
 */
#ifndef UTIL_FILE_H_
#define UTIL_FILE_H_

#include "obpmark.h"
#include "benchmark.h"

/* Definitions */
#define FILE_SUCCESS 	0
#define FILE_ERROR 	   -1

/* Functions */

int get_file_data(const char *file, const char *key, uint8_t *input_data, uint8_t *cypher_key, unsigned int data_length, unsigned int key_length);

#endif // UTIL_FILE_H_
