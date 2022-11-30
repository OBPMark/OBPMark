/**
 * \file util_prints.h 
 * \brief Benchmark #2.2 common print utils.
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
        // print implementation name
        printf("Implementation          = %s\n", IMPLEMENTATION_NAME);
        // print the command line arguments
        printf("Width                   = %d \n", print_info_data->w_size);
        printf("Height                  = %d \n", print_info_data->h_size);
        printf("Bit size                = %d \n", print_info_data->bit_size);
        printf("Segment size            = %d \n", print_info_data->segment_size);
        printf("Compression type        = ");
        if (print_info_data->type) {
            printf("Lossy\n");
        } else {
            printf("Lossless\n");
        }
        // check if block size is set
        #ifdef BLOCK_SIZE
        printf("block_size              = %d \n", BLOCK_SIZE);
        #endif
        #ifdef BLOCK_SIZE_PLANE
        printf("block_size_plane        = %d \n", BLOCK_SIZE_PLANE);
        #endif
        printf("\n");
        if(print_info_data->verbose_print)
        {
            printf("Arguments: \n");
            printf("csv_mode                = ");
            printf(BOOL_PRINT(print_info_data->csv_mode));
            printf("\n");
            printf("print_output            = ");
            printf(BOOL_PRINT(print_info_data->print_output));
            printf("\n");
            printf("database_mode           = ");
            printf(BOOL_PRINT(print_info_data->database_mode));
            printf("\n");
            printf("verbose_print           = ");
            printf(BOOL_PRINT(print_info_data->verbose_print));
            printf("\n");
            printf("input folder            = %s", print_info_data->input_file);
            printf("\n");
            printf("no_output_file          = ");
            printf(BOOL_PRINT(print_info_data->no_output_file));
            printf("\n");
            printf("output_file             = %s \n", print_info_data->output_file);
            printf("\n");
            
        }
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
     // generate benchmark metrics
    float total_time = host_to_device_time + execution_time + device_to_host_time;
    // calculate the throughput in MSamples/s
    float throughput = (print_info_data->w_size * print_info_data->h_size) / (total_time * 1000);

    if (print_info_data->csv_mode)
	{
        if (print_info_data->extended_csv_mode)
        {
            printf("%s;%s;%d;%d;%d;%d;%d;%.10f;%.10f;%.10f;%.10f;%.10f;%s;%s;\n", 
            BENCHMARK_ID, 
            IMPLEMENTATION_NAME, 
            print_info_data->w_size, 
            print_info_data->h_size, 
            print_info_data->bit_size, 
            print_info_data->segment_size,
            print_info_data->type,
            host_to_device_time, 
            execution_time, 
            device_to_host_time, 
            host_to_device_time + execution_time + device_to_host_time, 
            throughput,
            print_info_data->input_file, 
            print_info_data->output_file);
        }
        else
        {
            printf("%.10f;%.10f;%.10f;\n", host_to_device_time, execution_time, device_to_host_time);
        }
		 
		
	}
	else if (print_info_data->database_mode)
	{
		printf("%.10f;%.10f;%.10f;%ld;\n", host_to_device_time, execution_time, device_to_host_time, timestamp);
	}
	else
	{

        
        // print the metrics only in verbose mode
        printf("Benchmark metrics:\n");
        if(include_memory_transfer){
           printf("Elapsed time Host->Device = %.2f ms\n", host_to_device_time);
        }
		printf("Elapsed time execution = %.2f ms\n", execution_time );
        if (include_memory_transfer){
            printf("Elapsed time Device->Host = %.2f ms\n", device_to_host_time);
        }

        printf("Total execution time = %.2f ms\n", total_time);
        printf("Throughput = %.2f Mpixel/s\n", throughput);
        printf("\n");
	}
   


}
