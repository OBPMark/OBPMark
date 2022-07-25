/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #2.2
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "util_arg.h"
#include "obpmark_image.h"

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
	printf("usage: %s -w [size] -h [size] -b [size]\n", exec_name);
	printf(" -w size : width of the input image in pixels\n");
	printf(" -h size : height of the input image in pixels \n");
	printf(" -b size : bit size of each pixel\n");
	printf(" -s size : segment size\n");
	printf(" -y : type of compression, if this is argument is present the encoding will be using float, if not the codification is with integer\n");
	printf(" -f : input data file\n");
	printf(" -c : print time in CSV\n");
	printf(" -C : print time in CSV with timestamp\n");
	printf(" -t : print time in verbose\n");
	printf(" -o : print output\n");
}


int arguments_handler(int argc, char **argv, unsigned int *w_size, unsigned int *h_size, unsigned int *bit_size, unsigned int *segment_size, bool *type, bool *csv_mode, bool *database_mode, bool *print_output, bool *verbose_output, char *input_file)
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
			case 'b' : args +=1; *bit_size = atoi(argv[args]);break;
			case 'f' : args +=1; strcpy(input_file,argv[args]);break;
			case 's' : args +=1; *segment_size = atoi(argv[args]);break;
			case 'y' : *type = true;break;
			case 'c' : *csv_mode = true;break;
			case 'C' : *database_mode = true;break;
			case 'o' : *print_output = true;break;
			case 't' : *verbose_output = true;break;
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

	if((*bit_size > MAXIMUNBITSIZEFLOAT && !type) || (*bit_size > MAXIMUNBITSIZEINTEGER && type)) {
		printf("-b need to be set and smaller than or equal to %d\n\n", MAXIMUNBITSIZEFLOAT);
		print_usage(argv[0]);
		return ARG_ERROR;
	}

	if(*segment_size == 0) {
		// put default value for segment size
		*segment_size = DEFAULTSEGMENTSIZE;
	}
	else if (*segment_size < MINSEGMENTSIZE) {
		printf("-s need to be set and bigger than or equal to %d\n\n", MINSEGMENTSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}
	else if (*segment_size > MAXSEGMENTSIZE) {
		printf("-s need to be set and smaller than or equal to %d\n\n", MAXSEGMENTSIZE);
		print_usage(argv[0]);
		return ARG_ERROR;
	}
	
		

	return ARG_SUCCESS;
}
