/**
 * \file main_commandline.cpp
 * \author David Steenari (ESA)
 * \brief OBPMark Benchmark #3.1 AES Compression -- command line tool. 
 */
#include "obpmark.h"

#include <stdlib.h>
#include <stdio.h>

#include "aes.h"

#define EXIT_SUCCESS	0
#define ERROR_ARGUMENTS	-1
#define ERROR_FILELOAD	-2

#define OK_ARGUMENTS 	0


int arguments_handler(int argc, char **argv, unsigned int *num_iter, unsigned int *key_size, unsigned int *data_length, char *data_filepath, bool *csv_mode, bool *print_output);

// FIXME temporarly placement, move to other file 
int obpmark_aes(unsigned int key_size, uint8_t *data_buf, size_t buf_length)
{
	uint8_t *out_buf; 

	/* Encrypt */
	// FIXME start timer 
	//AES_encrypt(); // FIXME parameters
	// FIXME end timer

	return 0;
}

int exec_benchmark_aes(unsigned int num_iter, unsigned int key_size, unsigned int data_length, const char *data_filepath)
{
	int ret;
	FILE *fdata;
	uint8_t *data_buf; 
	size_t buf_length = 0;

	fdata = fopen(data_filepath, "rb");
	if(!fdata) {
		return ERROR_FILELOAD;
	}
	
	// FIXME add file reading 
	data_buf = 0; 

	// FIXME should be in separate file for automation "scripting"
	// FIXME file loading etc. functions should be separate from raw processing

	for(int i=0; i<num_iter; i++)
	{
		ret = obpmark_aes(key_size, data_buf, buf_length);  
		if(ret != EXIT_SUCCESS) {
			return ret;
		}
	}
	
	return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
	int ret;
	unsigned int num_iter=0, key_size=0, data_length=0;
	char *data_filepath = NULL; 
	bool csv_mode = false, print_output = false;

	/* Command line arguments processing */
	ret = arguments_handler(argc, argv, &num_iter, &key_size, &data_length, data_filepath, &csv_mode, &print_output);
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

void print_usage(const char *app_name)
{
	printf("Usage: %s -i <num_iter> -k <key_size> -l <data_length> <data_file>\n", app_name);
	printf(" data_file : path to file to encrypt\n");
	printf(" -i num_iter : number of iterations to execute\n");
	printf(" -k key_size : encryption key size\n");
	printf(" -l data_length : length of file to read in number of bytes (set to 0 to read full file)\n");
	printf(" -c : print time in CSV\n");
	printf(" -o : print output\n");
}

int arguments_handler(int argc, char **argv, unsigned int *num_iter, unsigned int *key_size, unsigned int *data_length, char *data_filepath, bool *csv_mode, bool *print_output)
{
	if(argc < 4) {
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	
	for(unsigned int args = 1; args < argc; ++args)
	{
		if(argv[args][0] != '-') {
			return ERROR_ARGUMENTS;
		}

		switch (argv[args][1]) {
			case 'i' : args +=1; *num_iter = atoi(argv[args]); break;
			case 'k' : args +=1; *key_size = atoi(argv[args]); break;
			case 'l' : args +=1; *data_length = atoi(argv[args]); break;
			case 'c' : *csv_mode = true; break;
			case 'o' : *print_output = true; break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}

	switch(*key_size) {
		case 128:
		case 192:
		case 256:
			break; 
		default:
			printf("error: key_size must be 128, 192 or 256\n");
			print_usage(argv[0]);
			return ERROR_ARGUMENTS;
	}
	// FIXME add other checks

	// FIXME cleanup
	/*if(*w_size < MINIMUNWSIZE) {
		printf("-w need to be set and bigger than or equal to %d\n\n", MINIMUNWSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if(*h_size < MINIMUNHSIZE) {
		printf("-h need to be set and bigger than or equal to %d\n\n", MINIMUNHSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if(*frames < MINIMUNFRAMES) {
		printf("-f need to be set and bigger than or equal to %d\n\n", MINIMUNFRAMES);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if(*bitsize != MINIMUNBITSIZE && *bitsize != MAXIMUNBITSIZE) {
		printf("-b need to be set and be %d or %d\n\n", MINIMUNBITSIZE, MAXIMUNBITSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}*/

	return OK_ARGUMENTS;
}
