/**
 * \file device.c
 * \brief Benchmark #121 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "device.h"
#include "processing.h"

void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	)
{
    init(image_data,t, 0,0, device_name);
}



void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	int platform,
	int device,
	char *device_name
	)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");

}



bool device_memory_init(
	compression_data_t *compression_data
	)
{	
// EMPTY
}



void copy_memory_to_device(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
// EMPTY
}


void process_benchmark(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{

unsigned int i;

for (i = 0; i < compression_data->step; ++i)
{
    preprocess_data();
	process_zeroblock();
    process_blocks();

}


}



void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{

}


void get_elapsed_time(
	compression_data_t *compression_data, 
	compression_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{

}


void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{

}