#include <time.h>
#include "benchmark_library.h"
#include "cpu_functions/cpu_functions.h"
#include <sys/time.h>

#define NUMBER_BASE 1
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	BenchmarkParameters *arguments_parameters = (BenchmarkParameters *)malloc(sizeof(BenchmarkParameters));

	int resolution = arguments_handler(argc,argv,arguments_parameters);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// A input vector
	int64_t size_A = arguments_parameters->size;
    int64_t mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);

	int64_t size_B = arguments_parameters->size;
    int64_t mem_size_B = sizeof(bench_t) * size_B;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	bench_t* d_B = (bench_t*) malloc(mem_size_B);
	bench_t aux_value = 0;
	// comparation result
	bool result = false;
	// strucs for CPU timing
	struct timespec start, end;
	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file) == 0)
	{
	// inicialice A matrix 
		for (int i=0; i<size_A; i++)
		{
			if (i % 2 == 0)
			{
				aux_value = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
			}	
			else
			{
				aux_value = 0;
			}
			if (arguments_parameters->print_input)
			{
				printf("%f ",aux_value);
			}
	    	A[i] = aux_value;
		}
		if (arguments_parameters->print_input)
		{
			printf("\n");
		}
	}
	else
	{	
		// load data
		get_double_hexadecimal_values(arguments_parameters->input_file, A,size_A);
	}
	
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////
	// copy A to B
	for (unsigned int i = 0; i < size_A; ++i)
	{
		h_B[i] = A[i];
		d_B[i] = A[i];
	}
	// base object init
	GraficObject *fft_bench = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(fft_bench, 0,arguments_parameters->gpu, device);
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}
	// init memory
	device_memory_init(fft_bench, size_B);
	// copy memory to device
	copy_memory_to_device(fft_bench, d_B, size_B);
	// execute kernel
	execute_kernel(fft_bench, arguments_parameters->size>>1);
	// copy memory to host
	copy_memory_to_host(fft_bench, d_B, size_B);

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(fft_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}
	if (arguments_parameters->print_output)
	{
		for (int i=0; i<size_B; i++){
	    	printf("%f ", d_B[i]);	
    		
		}
		// re print for get the same result of matlab
		for (int i=0; i<size_B; i++){
	    	printf("%f ", d_B[i]);	
    		
		}
		printf("\n");
	}
	


	if (arguments_parameters->verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		fft_function(h_B,arguments_parameters->size>>1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %lu milliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (arguments_parameters->print_output)
		{
			for (int i=0; i<size_B; i++){
		    	printf("%f ", h_B[i]);
			}
			// re print for get the same result of matlab
			for (int i=0; i<size_B; i++){
		    	printf("%f ", h_B[i]);
			}
			printf("\n");
		} 
	    result = compare_vectors(h_B, d_B, size_B);
	    if (result){
	    	printf("OK\n");
	    }
	    if (arguments_parameters->export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	    	print_double_hexadecimal_values(CPU_FILE, h_B, size_B);
	    }

	}
	if (arguments_parameters->export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(fft_bench);
	free(arguments_parameters);
	// free object memory 
	free(fft_bench);
	free(A);
	free(d_B);
	free(h_B);
return 0; 
}


// Arguments part

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}

void init_arguments(BenchmarkParameters* arguments_parameters){
	arguments_parameters->size = 0;
	arguments_parameters->gpu = 0;
	arguments_parameters->verification = false;
	arguments_parameters->export_results = false;
	arguments_parameters->export_results_gpu = false;
	arguments_parameters->print_output = false;
	arguments_parameters->print_input = false;
	arguments_parameters->print_timing = false;
	arguments_parameters->csv_format = false;
	arguments_parameters->mute_messages = false;
	arguments_parameters->csv_format_timestamp = false;
}

int arguments_handler(int argc, char ** argv, BenchmarkParameters* arguments_parameters){
	init_arguments(arguments_parameters);
	if (argc == 1){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			// comon part
			case 'v' : arguments_parameters->verification = true;break;
			case 'e' : arguments_parameters->verification = true; arguments_parameters->export_results= true;break;
			case 'o' : arguments_parameters->print_output = true;break;
			case 't' : arguments_parameters->print_timing = true;break;
			case 'c' : arguments_parameters->csv_format   = true;break;
			case 'C' : arguments_parameters->csv_format_timestamp = true;break;
			case 'g' : arguments_parameters->export_results_gpu = true;break;
			case 'd' : args +=1; arguments_parameters->gpu = atoi(argv[args]);break;
			case 'f' : arguments_parameters->mute_messages = true;break;
			// specific
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file,argv[args]);
					   break;
			case 's' : args +=1; arguments_parameters->size = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( arguments_parameters->size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}