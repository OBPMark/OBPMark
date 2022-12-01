/**
 * \file util_arg.h
 * \brief Command line argument util for Benchmark #1.2
 * \author Marc Sole Bonet (BSC)
 */
#ifndef UTIL_ARG_H_
#define UTIL_ARG_H_

#include "obpmark.h"
#include "benchmark.h"

#include <chrono>
#include <sys/time.h>
#include <ctime>

/* Definitions */
#define ARG_SUCCESS 	0
#define ARG_ERROR 	-1

/* Functions */

/**
 * \brief Command line argument handler.
 */
int arguments_handler(
        int argc, char **argv,
        unsigned int *in_height,
        unsigned int *in_width,
        unsigned int *ml_factor,
        bool *csv_mode,
        bool *database_mode,
        bool *print_output,
        bool *verbose_output,
        bool *random_data,
        bool *no_output_file,
        bool *extended_csv_mode,
        char *input_folder
        );

/**
 * \brief Prints command line usage. 
 */
void print_usage(const char *exec_name); 

/**
 * \brief gets the linux time in microseconds.
 */
long int get_timestamp();

#endif // UTIL_ARG_H_
