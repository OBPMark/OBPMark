/**
 * \file main.c 
 * \brief Benchmark #2.1 benchmark main file.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
// FIXME copy top comment+license from old code 
// FIXME add license to all files 

#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Optional utility headers */
#include "util_arg.h"
#include "util_data_rand.h"
#include "util_data_files.h"
#include "output_format_utils.h"


void print_output_result(struct OutputBitStream *output_bit_stream)
{
	printf("Total number of bytes: %u\n", output_bit_stream->num_total_bytes);
    for (unsigned int i = 0; i < output_bit_stream->num_total_bytes + 1; i++)
    {
        printf("%u, ", output_bit_stream->OutputBitStream[i]);
    }
    printf("\n");
}

void init_benchmark(
    unsigned int *input_data,
    struct OutputBitStream *output_bit_stream,

    unsigned int total_samples,
    unsigned int total_samples_step,
    unsigned int step,
    unsigned int n_bits,
    unsigned int j_blocksize,
    unsigned int r_samplesInterval,
    bool preprocessor_active,
    
    bool csv_mode, 
	bool print_output,
	bool database_mode,
	bool verbose_print,
	long int timestamp,
    bool debug_mode
	)
{

    compression_time_t *t = (compression_time_t *)malloc(sizeof(compression_time_t));
    compression_data_t *compression_data = (compression_data_t *)malloc(sizeof(compression_data_t));
    char device[100] = "";
	char* output_file = (char*)"output.bin";
    /* prepare compression data */
    compression_data->InputDataBlock = input_data;
    compression_data->OutputDataBlock = output_bit_stream;
    compression_data->n_bits = n_bits;
    compression_data->j_blocksize = j_blocksize;
    compression_data->r_samplesInterval = r_samplesInterval;
    compression_data->preprocessor_active = preprocessor_active;
    compression_data->TotalSamples = total_samples;
    compression_data->TotalSamplesStep = total_samples_step;
    compression_data->steps = step;
    compression_data->debug_mode = debug_mode;
    
    /* Device object init */
	init(compression_data, t, 0, DEVICESELECTED, device);
    

	if(!csv_mode && !database_mode){
		printf("Using device: %s\n", device);
	}
    /* Initialize memory on the device and copy data */
	device_memory_init(compression_data);
	copy_memory_to_device(compression_data, t);

	/* Run the benchmark, by processing the full frame list */
	process_benchmark(compression_data, t);

	/* Copy data back from device */
	copy_memory_to_host(compression_data, t);

	/* Get benchmark times */
	get_elapsed_time(compression_data, t, csv_mode, database_mode,verbose_print, timestamp);
	if(print_output)
	{
		print_output_result(output_bit_stream);
	}
	else 
	{
		// write the output image to a file call "output.bin"

		store_data_to_file(output_file, output_bit_stream->OutputBitStream, output_bit_stream->num_total_bytes + 1);
	}

	/* Clean and free device object */
	clean(compression_data, t);
}

int main(int argc, char **argv)
{

    // Seeding the rand algorithm
    srand(8111995);

	int ret;
    bool csv_mode = false;
	bool print_output = false;
	bool verbose_output = false;
	bool database_mode = false;
    bool debug_mode = false;
    int file_loading_output = 0;

    unsigned int steps = 0;
    unsigned int n_bits = 0;
    unsigned int j_blocksize = 0;
    unsigned int r_samplesInterval = 0;
    bool preprocessor_active = false;

    unsigned int *Input_data;
    struct OutputBitStream *Output_data;
    char input_file[100] = "";

    

    ret = arguments_handler(argc, argv, &steps, &n_bits,&j_blocksize, &r_samplesInterval, &preprocessor_active, &csv_mode, &database_mode, &print_output, &verbose_output, &debug_mode, input_file);
	if(ret == ARG_ERROR) {
		exit(-1);
	}

    const unsigned int TotalSamples = j_blocksize * r_samplesInterval; 
    const unsigned int TotalSamplesStep = TotalSamples * steps;
    /* Init Input_data */
    Input_data = ( unsigned int *)malloc(sizeof( unsigned int ) * TotalSamplesStep);
    /* Init Output_data */
    Output_data = (struct OutputBitStream *)malloc (sizeof(struct OutputBitStream));
    Output_data->OutputBitStream = (unsigned char *)calloc(TotalSamplesStep*4, sizeof(unsigned char));
    Output_data->num_bits = 0;
    Output_data->num_total_bytes = 0;

    /* Load data from files */
    file_loading_output = load_data_from_file(input_file, Input_data, j_blocksize, r_samplesInterval, steps);
    /* Init device and run test */
    init_benchmark(Input_data, Output_data, TotalSamples, TotalSamplesStep, steps, n_bits, j_blocksize, r_samplesInterval, preprocessor_active, csv_mode, print_output, database_mode, verbose_output, get_timestamp(), debug_mode);
    
    /* Free input data */
    free(Input_data);
    /* Free output data */
    free(Output_data->OutputBitStream);
    free(Output_data);

    return 0;
}