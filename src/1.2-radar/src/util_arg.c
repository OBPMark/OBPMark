/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #1.2
 * \author Marc Sole Bonet (BSC)
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
	printf("usage: %s -w [size] -h [size] -f [size]\n", exec_name);
	printf(" -w size : width of the input data in samples\n");
	printf(" -h size : height of the input data in samples\n");
	printf(" -m size : Multilook factor (number of width samples per pixel)\n");
	printf(" -r : random data\n");
	printf(" -F : location folder of the input data\n");
	printf(" -c : print time in CSV\n");
	printf(" -C : print time in CSV with timestamp\n");
	printf(" -t : print time in verbose\n");
	printf(" -o : print output\n");
}

int arguments_handler(int argc, char **argv, unsigned int *in_height, unsigned int *in_width, unsigned int *ml_factor, bool *csv_mode, bool *database_mode, bool *print_output, bool *verbose_output, bool *random_data, char *input_folder)
{
	if(argc < 4) {
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'w' : args +=1; *in_width = atoi(argv[args]);break;
			case 'h' : args +=1; *in_height = atoi(argv[args]);break;
			case 'm' : args +=1; *ml_factor = atoi(argv[args]);break;
			case 'F' : args +=1; strcpy(input_folder,argv[args]);break;
			case 'c' : *csv_mode = true;break;
			case 'C' : *database_mode = true;break;
			case 'r' : *random_data = true;break;
			case 'o' : *print_output = true;break;
			case 't' : *verbose_output = true;break;
			default: print_usage(argv[0]); return ARG_ERROR;
		}

	}

	if(*in_width < MINIMUNWSIZE) {
		printf("-w need to be set and bigger than or equal to %d\n\n", MINIMUNWSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	if(*in_height < MINIMUNHSIZE) {
		printf("-h need to be set and bigger than or equal to %d\n\n", MINIMUNHSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	if(*ml_factor <= 0){
		printf("-m need to be greater than 0\n\n");
		print_usage(argv[0]);
		return ARG_ERROR;
	}


	return ARG_SUCCESS;
}
