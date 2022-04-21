/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #3.1
 * \author Marc Solé Bonet (BSC)
 */
#include "util_arg.h"

#include <stdio.h>

long int get_timestamp(){
	struct timeval time_now{};
    gettimeofday(&time_now, nullptr);
    time_t msecs_time = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
	return (long int) msecs_time;
}

/* Functions */ 

void print_usage(const char *exec_name)
{
	printf("Usage: %s -k [size] -l [size]\n", exec_name);
	printf(" -l size : length of the plaintext to encrypt. Must be multiple of 16 \n");
	printf(" -k size : encryption key size (128, 192 or 256)\n");
	printf(" -m mode : execution mode (ctr or ecb). Default: ctr\n");
	printf(" -r : random data\n");
	printf(" -c : print time in CSV\n");
	printf(" -C : print time in CSV with timestamp\n");
	printf(" -t : print time in verbose\n");
	printf(" -o : print output\n");

//	printf("\n -f file : path to a file containing the cypher key\n");
}


int arguments_handler(int argc, char **argv, unsigned int *data_length, unsigned int *key_size, char **mode, bool *csv_mode, bool *database_mode, bool *print_output, bool *verbose_output, bool *random_data, char **key_filepath)
{
	if(argc < 2){
        print_usage(argv[0]); 
        return ARG_ERROR;
	}
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'l' : args +=1; *data_length = atoi(argv[args]); break;
			case 'k' : args +=1; *key_size = atoi(argv[args]); break;
            case 'm' : args +=1; *mode = argv[args]; break;
			case 'c' : *csv_mode = true;break;
			case 'C' : *database_mode = true;break;
			case 'r' : *random_data = true;break;
			case 'o' : *print_output = true;break;
			case 't' : *verbose_output = true;break;
 //           case 'f' : args +=1; *key_filepath = argv[args]; break;
			default: print_usage(argv[0]); return ARG_ERROR;
		}

	}

//	if(*data_length == 0){
//	    *data_length = std::filesystem::file_size(*data_filepath);
//	}
	
	//mandatory arguments
	if(*data_length < 0) {
		printf("-l need to be set and higher than 0\n\n");
	    print_usage(argv[0]);
	    return ARG_ERROR;
    }

	if(*mode == NULL){
        printf("error: -m mode must be either ctr or ecb\n\n");
	    print_usage(argv[0]);
	    return ARG_ERROR;
	}
    else {
        switch((unsigned int)(((*mode)[0]<<16) ^ ((*mode)[1]<<8) ^ ((*mode)[2]))) {
            case CONST_CTR:
            case CONST_ECB:
                break;
            default:
                printf("error: -m mode must be either ctr or ecb\n\n");
                print_usage(argv[0]);
                return ARG_ERROR;
        }
    }

    //validate values
	switch(*key_size) {
		case 128:
		case 192:
		case 256:
			break; 
		default:
			printf("error: -k must be set to 128, 192 or 256\n\n");
			print_usage(argv[0]);
			return ARG_ERROR;
	}
	if(*data_length%16!=0){
			printf("error: -l must be set to a multiple of 16\n\n");
			print_usage(argv[0]);
			return ARG_ERROR;
	}

	return ARG_SUCCESS;
}
