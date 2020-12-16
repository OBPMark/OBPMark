#include <time.h>
#include "benchmark_library.h"
#include "cpu/lib_cpu.h"
#include <sys/time.h>

#define NUMBER_BASE 1
// OUTPUT C is N x W matrix
// Print hexadecimal values of result 

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define GPU_FILE "gpu_file.out"
#define CPU_FILE "cpu_file.out"

int arguments_handler(int argc, char ** argv,int64_t *size, unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format, bool *validation_timing, bool *mute_messages, char *input_file_A, char *input_file_B);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	int64_t size = 0;
	unsigned int gpu = 0;
	bool verification  = false, export_results = false, print_output = false, print_timing = false, export_results_gpu = false, print_input = false, csv_format = false, validation_timing = false, mute_messages = false;
	char input_file_A[100] = "";
	char input_file_B[100] = "";

	int resolution = arguments_handler(argc,argv, &size, &gpu, &verification, &export_results, &export_results_gpu,&print_output, &print_timing, &csv_format, &validation_timing, &mute_messages, input_file_A, input_file_B);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// A input vector
	int64_t size_A = size;
    int64_t mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);

	int64_t size_B = size;
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
	if (strlen(input_file_A) == 0)
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
			if (print_input)
			{
				printf("%f ",aux_value);
			}
	    	A[i] = aux_value;
		}
		if (print_input)
		{
			printf("\n");
		}
	}
	else
	{	
		// load data
		get_double_hexadecimal_values(input_file_A, A,size_A);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE FOR ONLY TIMING  OF THE VALIDATION
	///////////////////////////////////////////////////////////////////////////////////////////////
	if(validation_timing){
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		fft_function(h_B,size>>1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (!mute_messages){
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		exit(0);
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
	GraficObject *fft_benck = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(fft_benck, 0,gpu, device);
	if (!csv_format && !mute_messages ){
		printf("Using device: %s\n", device);
	}
	// init memory
	device_memory_init(fft_benck, size_B);
	// copy memory to device
	copy_memory_to_device(fft_benck, d_B, size_B);
	// execute kernel
	execute_kernel(fft_benck, size>>1);
	// copy memory to host
	copy_memory_to_host(fft_benck, d_B, size_B);

	// get time
	if (print_timing || csv_format)
	{
		get_elapsed_time(fft_benck, csv_format);
	}
	if (print_output)
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
	


	if (verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		fft_function(h_B,size>>1);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (print_timing)
		{
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (print_output)
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
	    if (export_results){
	    	print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	    	print_double_hexadecimal_values(CPU_FILE, h_B, size_B);
	    }

	}
	if (export_results_gpu)
	{
		print_double_hexadecimal_values(GPU_FILE, d_B, size_B);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CLEAN MEMORY
	///////////////////////////////////////////////////////////////////////////////////////////////
	// clean device memory
	clean(fft_benck);
	// free object memory 
	free(fft_benck);
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
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verificaction of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -x: prints the timing of the validation. Only the sequential time of the application will be displayed\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}



int arguments_handler(int argc, char ** argv,int64_t *size, unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format, bool *validation_timing, bool *mute_messages, char *input_file_A, char *input_file_B){
	if (argc == 1){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			// comon part
			case 'v' : *verification = true;break;
			case 'e' : *verification = true; *export_results= true;break;
			case 'o' : *print_output = true;break;
			case 't' : *print_timing = true;break;
			case 'c' : *csv_format   = true;break;
			case 'g' : *export_results_gpu = true;break;
			case 'd' : args +=1; *gpu = atoi(argv[args]);break;
			case 'x' : *validation_timing = true;break;
			case 'f' : *mute_messages = true;break;
			// specific
			case 'i' : args +=1;
					   strcpy(input_file_A,argv[args]);
					   args +=1;
					   strcpy(input_file_B,argv[args]);
					   break;
			case 's' : args +=1; *size = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( *size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (*mute_messages){
		*csv_format = false;
	}
	return OK_ARGUMENTS;
}