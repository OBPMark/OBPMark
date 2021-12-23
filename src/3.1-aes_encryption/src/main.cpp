/**
 * \file main_commandline.cpp
 * \author David Steenari (ESA)
 * \brief OBPMark Benchmark #3.1 AES Compression -- command line tool. 
 */
#include "obpmark.h"

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <cstring>


#include "device.h"
#include "util_arg.h"
#include "util_file.h"

#define EXIT_SUCCESS	0
#define ERROR_ARGUMENTS -1


int exec_benchmark_aes(unsigned int num_iter, unsigned int key_size, unsigned int data_length, const char *data_filepath)
{
    uint8_t input[data_length];
    //set input value
    //FIXME Currently default value for validation
    for(int i=0; i<data_length; i++) input[i] = i*0x11;
    uint8_t key[(key_size/32)*4];
    //set key value
    //FIXME Currently default value for validation
    for(int i= 0; i<((key_size/32)*4); i++) key[i] = i;

    uint8_t out[data_length];

    uint8_t sbox[256] = SBOX_INIT;
	uint8_t rcon[11] = RCON_INIT;

    AES_time_t *t = (AES_time_t *)malloc(sizeof(AES_time_t));
    AES_data_t *AES_data =  (AES_data_t*) malloc(sizeof(AES_data_t));
    char device[100] = "";
    init(AES_data, t, device);

    //if(!csv_mode) 
    printf("Using device: %s\n", device);

    /* Initialize memory on the device and copy data */
    device_memory_init(AES_data, key_size, data_length);
    copy_memory_to_device(AES_data, key, input, sbox, rcon);

    /* Run the benchmark, by processing the full frame list */
    process_benchmark(AES_data, t);

    /* Copy data back from device */
    copy_memory_to_host(AES_data, out);
    puts("output:");
    for(int i = 0; i<data_length; i++) printf("%02x",out[i]);
    printf("\n");

    /* Get benchmark times */
    get_elapsed_time(t, 0);
//    if(print_output)
//    {
//        print_output_result(output_image);
//    }

    /* Clean and free device object */
    clean(AES_data, t);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned int num_iter=0, key_size=0, data_length=0;
	char *data_filepath = NULL; 
	bool csv_mode = false, print_output = false;

    
	/* Command line arguments processing */
	ret = arguments_handler(argc, argv, &num_iter, &key_size, &data_length, &data_filepath, &csv_mode, &print_output);
	if(ret == ERROR_ARGUMENTS) {
		return ERROR_ARGUMENTS;
	}

	/* Execute benchmark */
	ret = exec_benchmark_aes(num_iter, key_size, data_length, data_filepath);
	if(ret != EXIT_SUCCESS) {
		return ret;
	}

	return EXIT_SUCCESS;
}

