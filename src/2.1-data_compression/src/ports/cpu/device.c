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
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{
	if (csv_format)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;\n", (float) 0, elapsed_time, (float) 0);
	}
	else if (database_format)
	{
		
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;%ld;\n", (float) 0, elapsed_time, (float) 0, timestamp);
	}
	else if(verbose_print)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		//printf("Elapsed time Host->Device: %.10f ms\n", (float) 0);
		printf("Elapsed time kernel: %.10f ms\n", elapsed_time );
		//printf("Elapsed time Device->Host: %.10f ms\n", (float) 0);
	}

}


void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
	// free all memory from compression_data
	free(compression_data->OutputPreprocessedValue);


}