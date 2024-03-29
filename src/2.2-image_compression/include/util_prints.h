/**
 * \file util_prints.h 
 * \brief Benchmark #2.1  Input data generation.
 * \author Ivan Rodriguez (BSC)
 */

#ifndef UTIL_PRINTS_H_
#define UTIL_PRINTS_H_

#include "benchmark.h"

#define BENCHMARK_NAME "============ OBPMark #2.2 CCSDS 122.0 Image Compression ============" 
#define BENCHMARK_ID "2.2"
#define BOOL_PRINT(a) ((a) ? ("True"): ("False"))


struct print_info_data_t
{
	unsigned int w_size;
    unsigned int h_size;
    unsigned int bit_size;
    unsigned int segment_size;
    bool type;
    bool csv_mode;
    bool extended_csv_mode;
    bool print_output;
    bool database_mode;
    bool verbose_print;
    char *input_file;
    char *output_file;
    bool no_output_file;
};

void print_device_info(
    print_info_data_t *print_info_data,
    char *device_name
    );

void print_benchmark_info(
    print_info_data_t *print_info_data
    );


void print_execution_info(
    print_info_data_t *print_info_data,
    bool include_memory_transfer,
    long int timestamp,
    float host_to_device_time,
    float execution_time_dwt,
    float execution_time_bpe,
    float device_to_host_time
    );


#endif // UTIL_PRINTS_H_