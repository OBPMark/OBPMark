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

///**
// * \brief File open function
// * \param file_name Char array containing the path and file_name to open
// * \return The pointer to FILE object for the opened file, NULL if an error ocurred
// */
//FILE open_file(const char *file_name)

int get_file_data(const char *file, unsigned int length, uint8_t *buffer);

#endif // UTIL_FILE_H_
