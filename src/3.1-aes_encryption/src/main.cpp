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
#include "util_data_rand.c"

#define EXIT_SUCCESS	0
#define ERROR_ARGUMENTS -1

#define PRINT_CHAR 0

void print_data(uint8_t data[], unsigned int data_size){
#if PRINT_CHAR == 0
    printf("0x");
#endif
    for(int i = 0; i<data_size; i++) {
#if PRINT_CHAR == 1
        printf("%c",data[i]);
#else
        printf("%02x",data[i]);
#endif
    }
    printf("\n");
}

int exec_benchmark_aes(unsigned int num_iter, unsigned int key_size, unsigned int data_length, const char *data_filepath, bool csv_mode, bool print_output)
{
    uint8_t input[data_length];
    uint8_t key[key_size/8];
    uint8_t out[data_length];

    uint8_t sbox[256] = SBOX_INIT;
	uint8_t rcon[11] = RCON_INIT;

    AES_time_t *t = (AES_time_t *)malloc(sizeof(AES_time_t));
    AES_data_t *AES_data =  (AES_data_t*) malloc(sizeof(AES_data_t));
    char device[100] = "";

    if(!csv_mode) printf("Using device: %s\n", device);

    if(data_filepath==NULL) benchmark_gen_rand_data(input, key, data_length, key_size/8);
    else {
        int error = get_file_data(data_filepath, input, key, data_length, key_size/8);
        if(error < 0) return error;
    }

    if(print_output && data_filepath==NULL){
        printf("cypher key: ");
        print_data(key, key_size/8);
        printf("input text: ");
        print_data(input, data_length);
    }

    init(AES_data, t, device);

    /* Initialize memory on the device and copy data */
    device_memory_init(AES_data, key_size, data_length, num_iter);
    copy_memory_to_device(AES_data, key, input, sbox, rcon);

    /* Run the benchmark, by processing the full frame list */
    process_benchmark(AES_data, t);

    /* Copy data back from device */
    copy_memory_to_host(AES_data, out);

    /* Get benchmark times */
    get_elapsed_time(t, 0);
    if(print_output) 
        printf("encrypted text: ");
        print_data(out, data_length);

    /* Clean and free device object */
    clean(AES_data, t);

    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned int num_iter=1, key_size=-1, data_length=-1;
	char *data_filepath = NULL; 
	bool csv_mode = false, print_output = false;

    
	/* Command line arguments processing */
	ret = arguments_handler(argc, argv, &num_iter, &key_size, &data_length, &data_filepath, &csv_mode, &print_output);
	if(ret == ERROR_ARGUMENTS) {
		return ERROR_ARGUMENTS;
	}

	/* Execute benchmark */
	ret = exec_benchmark_aes(num_iter, key_size, data_length, data_filepath, csv_mode, print_output);
	if(ret != EXIT_SUCCESS) {
		return ret;
	}

	return EXIT_SUCCESS;
}

