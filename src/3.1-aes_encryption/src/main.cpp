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

//Print data has hexadecimal
void print_data(uint8_t data[], unsigned int data_size){
    printf("0x");
    for(int i = 0; i<data_size; i++) {
        printf("%02x",data[i]);
    }
}


void init_benchmark(
        uint8_t *input,
        uint8_t *key,
        uint8_t *iv,
        uint8_t *out,
        unsigned int data_length,
        unsigned int key_size,

        long int timestamp,
        print_info_data_t *benchmark_info
        )
{
    /* Init constant arrays */
    uint8_t sbox[256] = SBOX_INIT;
	uint8_t rcon[11] = RCON_INIT;

    /* Allocate memory for host data */
    AES_time_t *t = (AES_time_t *)malloc(sizeof(AES_time_t));
    AES_data_t *AES_data =  (AES_data_t*) malloc(sizeof(AES_data_t));

	char device[100] = "";
    
    // generate output filename that is output_<k_size>_<d_size>_IMPLEMENTATION_NAME_FILE.bin
    char output_filename[100] = "output_";
    char d_size_str[10];
    char k_size_str[10];
    // generate output filename
    sprintf(d_size_str, "%d", data_length);
    sprintf(k_size_str, "%d", key_size);
    strcat(output_filename, k_size_str);
    strcat(output_filename, "_");
    strcat(output_filename, d_size_str);
    strcat(output_filename, "_");
    strcat(output_filename, IMPLEMENTATION_NAME_FILE);
    strcat(output_filename, ".bin");
    char *output_file = (char*) output_filename;
    benchmark_info->output_file = output_file;
    


	init(AES_data, t, 0, DEVICESELECTED, device);

	print_device_info(benchmark_info, device);
	print_benchmark_info(benchmark_info);

    /* Initialize memory on the device and copy data */
    if(!device_memory_init(AES_data, key_size, data_length)){
	    printf("Error initializing memory\n");
        return;
    }
    copy_memory_to_device(AES_data, t, key, input, iv, sbox, rcon);

    /* Run the benchmark, by processing the full frame list */
    process_benchmark(AES_data, t);

    /* Copy data back from device */
    copy_memory_to_host(AES_data, t, out);

    /* Get benchmark times */
	get_elapsed_time(t, benchmark_info, timestamp);

	if(benchmark_info->print_output)
    {
        print_data(out, data_length);
    }
	else 
    {
		// write the output image to a file call "output.bin"
        if (!benchmark_info->no_output_file)
        {
            frame8_t *output_frame = (frame8_t*) malloc(sizeof(frame8_t));
            output_frame->w = data_length;
            output_frame->h = 1;
            output_frame->f = out;
            int result = write_frame8(output_file, output_frame, 0);
			if (result == 1 && !benchmark_info->csv_mode && !benchmark_info->database_mode)
			{
				printf("Done. Outputs written to %s\n", output_file);
			}
		}
    }

    /* Clean and free device object */
    clean(AES_data, t);
}

int main(int argc, char **argv)
{
	int ret;
	unsigned int key_size=-1, data_length=-1;
	char input_folder[100] = ""; 
	bool csv_mode = false;
	bool database_mode = false;
	bool print_output = false;
	bool verbose_output = false;
	bool random_data = false;
	bool no_output_file = false;
	bool extended_csv_mode = false;

    uint8_t *input;
    uint8_t *key;
    uint8_t *iv;
    uint8_t *out;
    
	/* Command line arguments processing */
	ret = arguments_handler(argc, argv, &data_length, &key_size, &csv_mode, &database_mode, &print_output, &verbose_output, &random_data, &no_output_file, &extended_csv_mode, input_folder);
	if(ret == ARG_ERROR) {
        exit(-1);
	}
	
    // If no folder is set, use default
    if(strcmp(input_folder,"") == 0)
        sprintf(input_folder, "%s/%d", DEFAULT_INPUT_FOLDER, key_size);
    // If random data, no folder is used
	if(random_data) input_folder[0]=0;

	/*Set benchmark info data */
	print_info_data_t *benchmark_info = (print_info_data_t *)malloc(sizeof(print_info_data_t));
	benchmark_info -> d_size = data_length;
	benchmark_info -> k_size = key_size;
	benchmark_info -> Nb = 4;
	benchmark_info -> Nk = key_size/8/4;
	benchmark_info -> Nr = benchmark_info->Nk + 6;
	benchmark_info -> csv_mode = csv_mode;
	benchmark_info -> print_output = print_output;
	benchmark_info -> database_mode = database_mode;
	benchmark_info -> verbose_print = verbose_output;
	benchmark_info -> random_data = random_data;
	benchmark_info -> no_output_file = no_output_file;
	benchmark_info -> input_folder = input_folder;
	benchmark_info -> extended_csv_mode = extended_csv_mode;


    /* Allocate input data */
    input = (uint8_t*) malloc(sizeof(uint8_t)*data_length);
    key = (uint8_t*) malloc(sizeof(uint8_t)*key_size/8);
    iv = (uint8_t*) malloc(sizeof(uint8_t)*16);

    /* Allocate input data */
    out = (uint8_t*) malloc(sizeof(uint8_t)*data_length);

    if (random_data) benchmark_gen_rand_data(input, key, iv, data_length, key_size/8);
    else if(load_data_from_files(input, key, iv, data_length, key_size, input_folder) == FILE_LOADING_ERROR) exit(-1);

	/* Init device and run test */
	init_benchmark(input, key, iv, out, data_length, key_size, get_timestamp(), benchmark_info);

	free(input);
	free(key);
	free(iv);
	free(out);
	free(benchmark_info);

	return 0;
}

