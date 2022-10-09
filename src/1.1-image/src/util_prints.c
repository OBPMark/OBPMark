/**
 * \file util_prints.h 
 * \brief Benchmark #1.1 common print utils.
 * \author Ivan Rodriguez (BSC)
 */

#include "util_prints.h"

void print_device_info(
    print_info_data_t *print_info_data,
    char *device_name
    )
{
    if(!print_info_data->csv_mode && !print_info_data->database_mode){
		printf("Using device: %s\n", device_name);
	}
}


void print_benchmark_info(
    print_info_data_t *print_info_data
    )
{
    if(!print_info_data->csv_mode && !print_info_data->database_mode){
        printf(BENCHMARK_NAME);
        printf("\n");
        // print the command line arguments
        printf("Arguments: ");
        printf("width =%d ", print_info_data->w_size);
        printf("height =%d ", print_info_data->h_size);
        printf("num_frames =%d ", print_info_data->num_frames);
        printf("csv_mode =%d ", print_info_data->csv_mode);
        printf("print_output = ");
        printf(BOOL_PRINT(print_info_data->print_output));
        printf(" ");
        printf("database_mode = ");
        printf(BOOL_PRINT(print_info_data->database_mode));
        printf(" ");
        printf("verbose_print = ");
        printf(BOOL_PRINT(print_info_data->verbose_print));
        printf(" ");
        printf("random_data = ");
        printf(BOOL_PRINT(print_info_data->random_data));
        printf(" ");
        printf("no_output_file = ");
        printf(BOOL_PRINT(print_info_data->no_output_file));
        printf("\n");
    
    }
    

}


void print_execution_info(
    print_info_data_t *print_info_data,
    bool include_memory_transfer,
    long int timestamp,
    float host_to_device_time,
    float execution_time,
    float device_to_host_time
    )
{

    if (print_info_data->csv_mode)
	{
		 
		printf("%.10f;%.10f;%.10f;\n", host_to_device_time, execution_time, device_to_host_time);
	}
	else if (print_info_data->database_mode)
	{
		printf("%.10f;%.10f;%.10f;%ld;\n", host_to_device_time, execution_time, device_to_host_time, timestamp);
	}
	else if(print_info_data->verbose_print)
	{
        if(include_memory_transfer){
           printf("Elapsed time Host->Device: %.2f ms\n", host_to_device_time);
        }
		printf("Elapsed time execution: %.2f ms\n", execution_time );
        if (include_memory_transfer){
            printf("Elapsed time Device->Host: %.2f ms\n", device_to_host_time);
        }
	}
    // generate benchmark metrics
    float total_time = host_to_device_time + execution_time + device_to_host_time;
    float total_time_per_frame = total_time / print_info_data->num_frames;
    // calculate the throughput in Mpixel/s
    float throughput = (print_info_data->w_size * print_info_data->h_size) / (total_time * 1000);
    // print the metrics only in verbose mode
    if(print_info_data->verbose_print){
        printf("Benchmark metrics:\n");
        printf("Total time: %.2f ms\n", total_time);
        printf("Total time per frame (average): %.2f ms\n", total_time_per_frame);
        printf("Throughput: %.2f Mpixel/s\n", throughput);
    }


}
