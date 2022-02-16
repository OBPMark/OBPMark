/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #3.1
 * \author Marc Solé Bonet (BSC)
 */
#include "util_arg.h"

#include <stdio.h>
#include <filesystem>

/* Functions */ 

void print_usage(const char *exec_name)
{
	printf("Usage: %s -k [key_size] -l [data_length]\n", exec_name);
	printf(" -k key_size : encryption key size (128, 192 or 256)\n");
	printf(" -y key_file : path to file containing the cypher key\n");
	printf(" -l data_length : length of the data to encrypt in bytes. Must be multiple of 16 (set to 0 to read full file)\n");
	//printf(" -l data_length : length of file to read in number of bytes (set to 0 to read full file)\n");
	printf(" -f data_file : path to file to encrypt\n");
	//printf(" -c : print time in CSV\n");
	printf(" -i : print random generated input\n");
	printf(" -o : print output\n");
	printf(" -s : does not print the time");
}

int arguments_handler(int argc, char **argv, unsigned int *key_size, char **key_filepath, unsigned int *data_length, char **data_filepath, bool *csv_mode, bool *print_output, bool *print_input, bool *silent)
{

	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'k' : args +=1; *key_size = atoi(argv[args]); break;
            case 'y' : args +=1; *key_filepath = argv[args]; break;
			case 'l' : args +=1; *data_length = atoi(argv[args]); break;
            case 'f' : args +=1; *data_filepath = argv[args]; break;
			case 'c' : *csv_mode = true; break;
			case 'o' : *print_output = true; break;
			case 'i' : *print_input = true; break;
			case 's' : *silent = true; break;
			default: print_usage(argv[0]); return ARG_ERROR;
		}

	}

	if(*data_length == 0){
	    *data_length = std::filesystem::file_size(*data_filepath);
	}
	
	//mandatory arguments
	if(*key_size < 0 || *data_length < 0) {
	    print_usage(argv[0]);
	    return ARG_ERROR;
	}

    //validate values
	switch(*key_size) {
		case 128:
		case 192:
		case 256:
			break; 
		default:
			printf("error: key_size must be 128, 192 or 256\n");
			print_usage(argv[0]);
			return ARG_ERROR;
	}
	if(*data_length%16!=0){
			printf("error: data_length must be multiple of 16\n");
			print_usage(argv[0]);
			return ARG_ERROR;
	}

	return ARG_SUCCESS;
}
