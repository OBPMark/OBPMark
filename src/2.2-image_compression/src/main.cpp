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


void print_output_result()
{

}

void init_benchmark(compression_image_data_t* ccsds_data,compression_time_t* t ,bool csv_mode, bool database_mode, bool print_output, bool verbose_print, long int timestamp)
{
    
    char device[100] = "";

    /* Device object init */
	init(ccsds_data, t, 0, DEVICESELECTED, device);

	if(!csv_mode && !database_mode){
		printf("Using device: %s\n", device);
	}

    // init memory in the device
    device_memory_init(ccsds_data);
    copy_memory_to_device(ccsds_data, t);
    // encode image
    process_benchmark(ccsds_data, t);

    // copy memory from device to host
    copy_memory_to_host(ccsds_data, t);
    
    // get elapsed time
    get_elapsed_time(ccsds_data, t, csv_mode, database_mode, verbose_print, timestamp);



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

    unsigned int w_size = 0;
    unsigned int h_size = 0;
    unsigned int bit_size = 0;
    unsigned int segment_size = 0;

    int file_loading_output = 0;
    int file_storage_output = 0;

    unsigned int pad_rows = 0;
    unsigned int pad_columns = 0;

    char input_file[100] = "";
    char output_file[100] = "";

    ret = arguments_handler(argc, argv, &w_size, &h_size, &bit_size, &segment_size,&type, &csv_mode, &database_mode, &print_output, &verbose_output, input_file, output_file);
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

    init_benchmark(ccsds_data, t, csv_mode, database_mode, print_output, verbose_output,get_timestamp());
    // write the output file
    write_segment_list(ccsds_data->segment_list, ccsds_data->number_of_segments ,output_file);
    // free memory
    clean(ccsds_data, t);

}