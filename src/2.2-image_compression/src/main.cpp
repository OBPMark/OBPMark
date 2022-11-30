/**
 * \file main.c 
 * \brief Benchmark #2.2 benchmark main file.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
// FIXME copy top comment+license from old code 
// FIXME add license to all files 
#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Optional utility headers */
#include "util_arg.h"
#include "util_data_files.h"


void init_benchmark(compression_image_data_t* ccsds_data,compression_time_t* t ,print_info_data_t *benchmark_info, long int timestamp)
{
    
    char device[100] = "";
    char output_filename[100] = "output_";
	char segment_size[10];
	char w_size_str[10];
    sprintf(w_size_str, "%d", benchmark_info->w_size); // NOTE: -4 because of the 4 extra frames that are needed for computation
	sprintf(segment_size, "%d", benchmark_info->segment_size);
	strcat(output_filename, w_size_str);
	strcat(output_filename, "_");
	strcat(output_filename, segment_size);
	strcat(output_filename, "_");
	strcat(output_filename, IMPLEMENTATION_NAME_FILE);
	strcat(output_filename, ".bin");
	char* output_file = (char*)output_filename;
    benchmark_info->output_file = output_file;

    /* Device object init */
	init(ccsds_data, t, 0, DEVICESELECTED, device);

	print_device_info(benchmark_info, device);
	print_benchmark_info(benchmark_info);

    // init memory in the device
    device_memory_init(ccsds_data);
    copy_memory_to_device(ccsds_data, t);
    // encode image
    process_benchmark(ccsds_data, t);

    // copy memory from device to host
    copy_memory_to_host(ccsds_data, t);
    
    // get elapsed time
    get_elapsed_time(ccsds_data, t, benchmark_info, timestamp);

    // write the output image to a file call "output.bin"
    if (!benchmark_info->no_output_file)
    {
        // write the output file
        write_segment_list(ccsds_data->segment_list, ccsds_data->number_of_segments ,output_file);
        if (!benchmark_info->csv_mode && !benchmark_info->database_mode)
        {
            printf("Done. Outputs written to %s\n", output_file);
        }
    }
		




}


int main(int argc, char **argv)
{
    // Seeding the rand algorithm
    srand(21121993);

    int ret;
    bool csv_mode = false;
	bool print_output = false;
	bool verbose_output = false;
	bool database_mode = false;
    bool type = false; // If type is false we will perform the encode with integer else will perform with float
    bool no_output_file = false;
	bool extended_csv_mode = false;

    unsigned int w_size = 0;
    unsigned int h_size = 0;
    unsigned int bit_size = 0;
    unsigned int segment_size = 0;

    int file_loading_output = 0;
    int file_storage_output = 0;

    unsigned int pad_rows = 0;
    unsigned int pad_columns = 0;

    char input_file[100] = "";


    ret = arguments_handler(argc, argv, &w_size, &h_size, &bit_size, &segment_size,&type, &csv_mode, &database_mode, &print_output, &verbose_output,&no_output_file, &extended_csv_mode, input_file);
    if(ret == ARG_ERROR) {
		exit(-1);
	}

    compression_image_data_t *ccsds_data = (compression_image_data_t *)malloc(sizeof(compression_image_data_t));
    compression_time_t *t = (compression_time_t *)malloc(sizeof(compression_time_t));
    ccsds_data->w_size = w_size;
    ccsds_data->h_size = h_size;
    ccsds_data->bit_size = bit_size;
    ccsds_data->segment_size = segment_size;
    ccsds_data->type_of_compression = type;

    /* Init print information data */
	print_info_data_t *benchmark_info = (print_info_data_t *)malloc(sizeof(print_info_data_t));
	benchmark_info -> w_size = w_size;
	benchmark_info -> h_size = h_size;
    benchmark_info -> bit_size = bit_size;
	benchmark_info -> segment_size = segment_size;
    benchmark_info -> type = type;
	benchmark_info -> csv_mode = csv_mode;
	benchmark_info -> print_output = print_output;
	benchmark_info -> database_mode = database_mode;
	benchmark_info -> verbose_print = verbose_output;
	benchmark_info -> no_output_file = no_output_file;
	benchmark_info -> input_file = input_file;
	benchmark_info -> extended_csv_mode = extended_csv_mode;



    if(ccsds_data->h_size % BLOCKSIZEIMAGE != 0){
		pad_rows =  BLOCKSIZEIMAGE - (ccsds_data->h_size % BLOCKSIZEIMAGE);
    }
        
    if(ccsds_data->w_size % BLOCKSIZEIMAGE != 0){
        pad_columns = BLOCKSIZEIMAGE - (ccsds_data->w_size % BLOCKSIZEIMAGE);
    }
    ccsds_data->pad_rows = pad_rows;
    ccsds_data->pad_columns = pad_columns;
    // init input image with the padding
    ccsds_data->input_image = (int*)calloc((ccsds_data->h_size + ccsds_data->pad_rows) * (ccsds_data->w_size + ccsds_data->pad_columns), sizeof(int));

    // load the image
    file_loading_output = load_data_from_file(input_file, ccsds_data);
    if (file_loading_output == FILE_LOADING_ERROR)
    {
        exit(-1);
    }

    init_benchmark(ccsds_data, t, benchmark_info ,get_timestamp());
    
    
    // free memory
    clean(ccsds_data, t);
    free(benchmark_info);

}