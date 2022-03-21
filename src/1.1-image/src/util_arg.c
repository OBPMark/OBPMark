/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #1.1
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "util_arg.h"

#include <stdio.h>

/* Functions */ 

void print_usage(const char *exec_name)
{
	printf("usage: %s -w [size] -h [size] -f [size]\n", exec_name);
	printf(" -f size : number of frames\n");
	printf(" -w size : width of the input image in pixels\n");
	printf(" -h size : height of the input image in pixels \n");
	printf(" -r : random data\n");
	printf(" -c : print time in CSV\n");
	printf(" -t : print time in full\n");
	printf(" -o : print output\n");
}

int arguments_handler(int argc, char **argv, unsigned int *w_size, unsigned int *h_size, unsigned int *frames, bool *csv_mode, bool *print_output, bool *full_time_output, bool *random_data)
{
	if(argc < 3) {
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'w' : args +=1; *w_size = atoi(argv[args]);break;
			case 'h' : args +=1; *h_size = atoi(argv[args]);break;
			case 'f' : args +=1; *frames = atoi(argv[args]);break;
			case 'c' : *csv_mode = true;break;
			case 'r' : *random_data = true;break;
			case 'o' : *print_output = true;break;
			case 't' : *full_time_output = true;break;
			default: print_usage(argv[0]); return ARG_ERROR;
		}

	}

	if(*w_size < MINIMUNWSIZE) {
		printf("-w need to be set and bigger than or equal to %d\n\n", MINIMUNWSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	if(*h_size < MINIMUNHSIZE) {
		printf("-h need to be set and bigger than or equal to %d\n\n", MINIMUNHSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	if(*frames < MINIMUNFRAMES) {
		printf("-f need to be set and bigger than or equal to %d\n\n", MINIMUNFRAMES);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	return ARG_SUCCESS;
}
