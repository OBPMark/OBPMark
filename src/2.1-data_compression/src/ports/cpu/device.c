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
    init(compression_data,t, 0,0, device_name);
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
	// do a calloc  OutputPreprocessedValue with size totalSamples
	compression_data->OutputPreprocessedValue = (unsigned int *) calloc(compression_data->TotalSamples, sizeof(unsigned int));
	return true;
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

T_START(t->t_test);
unsigned int step;
//create a arrat of ZeroBlockCounter with size r_samplesInterval
struct ZeroBlockCounter *ZeroCounter = (ZeroBlockCounter *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockCounter));
struct ZeroBlockProcessed *ZBProcessed= (ZeroBlockProcessed *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockProcessed));





for (step = 0; step < compression_data->steps; ++step)
{
	if(compression_data->debug_mode){printf("Step %d\n",step);}
    unsigned int ZeroCounterPos = 0;
	// for each step init zero counter
	for(int i = 0; i < compression_data->r_samplesInterval; ++i) { ZeroCounter[i].counter = 0; ZeroCounter[i].position = -1; }
    preprocess_data(compression_data,&ZeroCounterPos,ZeroCounter, step);
	// ZeroBlock processed array per position
	for(int i = 0; i < compression_data->r_samplesInterval; ++i) { ZBProcessed[i].NumberOfZeros = -1; }
	process_zeroblock(compression_data,&ZeroCounterPos,ZeroCounter,ZBProcessed);
	// Compressing each block
	process_blocks(compression_data, ZBProcessed, step);

}

T_STOP(t->t_test);
// free ZeroCounter
free(ZeroCounter);
free(ZBProcessed);

}



void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
//EMPTY
}



void get_elapsed_time(
	compression_data_t *compression_data, 
	compression_time_t *t, 
	print_info_data_t *benchmark_info,
	long int timestamp
	)
{	
	double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000));
	print_execution_info(benchmark_info, false, timestamp,0,(float)(elapsed_time),0);
}



void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
	// free all memory from compression_data
	free(compression_data->OutputPreprocessedValue);


}