/**
 * \file main.cpp
 * \author Marc Sole Bonet (BSC)
 * \brief OBPMark Benchmark #3.1 AES Compression main file.
 */
#include "obpmark.h"

#include "benchmark.h"
#include "device.h"

/* Utility headers */
#include "util_arg.h"
#include "util_data_files.h"
#include "util_data_rand.h"

#define EXIT_SUCCESS	0
#define ERROR_ARGUMENTS -1
#define ERROR_INIT -2


//Print data has hexadecimal
void print_data(uint8_t data[], unsigned int data_size){
    printf("0x");
    for(int i = 0; i<data_size; i++) {
        printf("%02x",data[i]);
    }
}


int exec_benchmark_aes(unsigned int data_length, unsigned int key_size, char *mode,  bool csv_mode, bool database_mode, bool print_output, bool verbose_output, bool random_data, const char *key_filepath )
{
    char device[100] = "";
    char *output_file = (char*)"output.bin";
	int file_loading_output = 0;
    long int timestamp = get_timestamp();

    /* Allocate memory for input values */
    uint8_t *input;
    input = (uint8_t*) malloc(sizeof(uint8_t)*data_length);
    uint8_t *key;
    key = (uint8_t*) malloc(sizeof(uint8_t)*key_size/8);
    uint8_t *iv;
    iv = (uint8_t*) malloc(sizeof(uint8_t)*16);
    uint8_t *out;
    out = (uint8_t*) malloc(sizeof(uint8_t)*data_length);


    uint8_t sbox[256] = SBOX_INIT;
	uint8_t rcon[11] = RCON_INIT;

    /* Allocate memory for host data */
    AES_time_t *t = (AES_time_t *)malloc(sizeof(AES_time_t));
    AES_data_t *AES_data =  (AES_data_t*) malloc(sizeof(AES_data_t));
    AES_mode_t enc_mode;
	switch((unsigned int)((mode[0]<<16) ^ (mode[1]<<8) ^ (mode[2]))) {
        case CONST_CTR:
            enc_mode = AES_CTR;
            break;
        case CONST_ECB:
            enc_mode = AES_ECB;
            break;
        default:
            enc_mode = AES_CTR;
            break;
	}


    if (random_data)
    {
        benchmark_gen_rand_data(input, key, iv, data_length, key_size/8);
    }
    else 
    {
		/* Load data from files */
		
		file_loading_output = load_data_from_files(
		input, key, iv, data_length, key_size);
		if (file_loading_output == FILE_LOADING_ERROR)
		{
			exit(-1);
		}
    }

    init(AES_data, t, device);
    if(!csv_mode) printf("Using device: %s\n", device);

    /* Initialize memory on the device and copy data */
    if(!device_memory_init(AES_data, enc_mode, key_size, data_length)){
        return ERROR_INIT;
    }
    copy_memory_to_device(AES_data, t, key, input, iv, sbox, rcon);

    /* Run the benchmark, by processing the full frame list */
    process_benchmark(AES_data, t);

    /* Copy data back from device */
    copy_memory_to_host(AES_data, t, out);

    /* Get benchmark times */
	get_elapsed_time(t, csv_mode, database_mode, verbose_output, timestamp);
    if(print_output) {
        printf("encrypted text: ");
        print_data(out, data_length);
        printf("\n");
    }
    else{
        frame8_t *output_frame = (frame8_t*) malloc(sizeof(frame8_t));
        output_frame->w = data_length;
        output_frame->h = 1;
        output_frame->f = out;
        write_frame8 (output_file, output_frame, 1);
    }

    /* Clean and free device object */
    clean(AES_data, t);
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned int key_size=-1, data_length=-1;
	char *key_filepath = NULL; 
	char *enc_mode = (char*)"ctr";
	bool csv_mode = false, database_mode = false, print_output = false, verbose_output = false, random_data = false;

    
	/* Command line arguments processing */
	ret = arguments_handler(argc, argv, &data_length, &key_size, &enc_mode, &csv_mode, &database_mode, &print_output, &verbose_output, &random_data, &key_filepath);
	if(ret == ERROR_ARGUMENTS) {
		return ERROR_ARGUMENTS;
	}
	

	/* Execute benchmark */
	ret = exec_benchmark_aes(data_length, key_size, enc_mode, csv_mode, database_mode, print_output, verbose_output, random_data, key_filepath);
	if(ret != EXIT_SUCCESS) {
		return ret;
	}

	return EXIT_SUCCESS;
}

