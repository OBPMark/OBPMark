/**
 * \file util_arg.c
 * \brief Command line argument util for Benchmark #2.1
 * \author Ivan Rodriguez-Ferrandez (BSC)
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
	printf("usage: %s -s [size] -n [size] -j [size] -r [size]\n", exec_name);
	printf(" -s size : number of samples\n");
	printf(" -n size : number of bits\n");
	printf(" -r size : size of the interval sample \n");
	printf(" -j size : block size\n");
    printf(" -p : preprocessor_active\n");
	printf(" -f : file that contains input data\n");
	printf(" -c : print time in CSV\n");
	printf(" -C : print time in CSV with timestamp\n");
	printf(" -E : extended csv output\n");
	printf(" -v : verbose print\n");
	printf(" -t : no generate output file\n");
	printf(" -o : print output\n");
}

int arguments_handler(
	int argc,
	char **argv,
	unsigned int *steps,
	unsigned int *n_bits,
    unsigned int *j_blocksize,
	unsigned int *r_samplesInterval,
    bool *preprocessor_active,
	bool *csv_mode,
	bool *database_mode,
	bool *print_output,
	bool *verbose_output,
	bool *debug_mode,
	bool *no_output_file,
	bool *extended_csv_mode,
	char *input_file
	)

{
	if(argc < 3) {
		print_usage(argv[0]);
		return ARG_ERROR;
	}
	unsigned int size = 0;
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 's' : args +=1; size = atoi(argv[args]);break;
			case 'n' : args +=1; *n_bits = atoi(argv[args]);break;
			case 'r' : args +=1; *r_samplesInterval = atoi(argv[args]);break;
            case 'f' : args +=1; strcpy(input_file,argv[args]);break;
            case 'j' : args +=1; *j_blocksize = atoi(argv[args]);break;
            case 'p' : *preprocessor_active = true;break;
			case 'c' : *csv_mode = true;break;
			case 'E' : *extended_csv_mode = true;break;
			case 'D' : *debug_mode = true;break;
			case 'C' : *database_mode = true;break;
			case 'o' : *print_output = true;break;
			case 'v' : *verbose_output = true;break;
			case 't' : *no_output_file = true;break;
			default: print_usage(argv[0]); return ARG_ERROR;
		}

	}

    // Ensures that the config parameters are set correctly
    // Steps must be a positive number greater than 0

	// check size to define number of steps
	if (size < *r_samplesInterval)
	{
		printf("Error: size must be greater than Samples Interval\n");
		return ARG_ERROR;
	}
	// check size can be divided by samples interval without remainder
	if (size % *r_samplesInterval != 0)
	{
		printf("Error: size must be divisible by Samples Interval\n");
		return ARG_ERROR;
	}
	// update number of steps
	*steps = size / (*r_samplesInterval * (*j_blocksize));
	
    // n_bits must be a positive number greater than 0 and not greater than 32
    if(*n_bits <= MINIMUMBITSIZE || *n_bits > MAXIMUNBITSIZE)
    {
        printf("error: n_bits must be a positive number greater than 0 and not greater than 32\n");
        return ARG_ERROR;
    }
    // blocksize must be 8 or 16 or 32 or 64
    if(*j_blocksize != JBLOCKSIZE1 && *j_blocksize != JBLOCKSIZE2 && *j_blocksize != JBLOCKSIZE3 && *j_blocksize != JBLOCKSIZE4)
    {
        printf("error: blocksize must be 8 or 16 or 32 or 64\n");
        return ARG_ERROR;
    }
    // r_samplesInterval must be a positive number greater than 0 and less that MAX_NUMBER_OF_BLOCKS
	if(*r_samplesInterval <= MINIMUMSAMPLINGINTERVAL || *r_samplesInterval > MAX_NUMBER_OF_BLOCKS)
	{
		printf("error: Samples Interval must be a positive number greater than 0 and less that %d\n", MAX_NUMBER_OF_BLOCKS);
		return ARG_ERROR;
	}

	// input_file must be a valid file
   
    FILE *file = fopen(input_file, "r");
    if(file == nullptr)
    {
        printf("error: input_file must be a valid file\n");
        return ARG_ERROR;
    }
    fclose(file);
	
	// if extended csv mode is enabled, csv mode is enabled too
	if(*extended_csv_mode) *csv_mode = true;
    

	return ARG_SUCCESS;
}
