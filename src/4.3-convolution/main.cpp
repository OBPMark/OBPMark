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
	// linearizable versions of matrix
	unsigned int size_matrix =arguments_parameters->size * arguments_parameters->size;
	// A input matrix
	unsigned int size_A = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B input matrix
	unsigned int size_B = arguments_parameters->size * arguments_parameters->size;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	bench_t* d_B = (bench_t*) malloc(mem_size_B);
	// kernel matrix
	unsigned int size_k = arguments_parameters->kernel_size * arguments_parameters->kernel_size ;
    unsigned int mem_size_k = sizeof(bench_t) * size_k;
	bench_t* kernel = (bench_t*) malloc(mem_size_k);
	// comparation result
	bool result = false;
	// strucs for CPU timing
	struct timespec start, end;
	///////////////////////////////////////////////////////////////////////////////////////////////
	// DATA INIT
	///////////////////////////////////////////////////////////////////////////////////////////////
	if (strlen(arguments_parameters->input_file_A) == 0)
	{
	// inicialice A matrix 
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	    		#ifdef INT
	        	A[i*arguments_parameters->size+j] = rand() % (NUMBER_BASE * 100);

	        	#else
	        	A[i*arguments_parameters->size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	#endif
	    	}
		}
	// iniciate B matrix 
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	        	h_B[i*arguments_parameters->size+j] = 0;
	        	d_B[i*arguments_parameters->size+j] = 0;
	    	}
		}
	// iniciate kernel matrix
		for (int i=0; i < size_k; ++i)
		{
			#ifdef INT
	        	kernel[i] =  rand() % (NUMBER_BASE * 100);
	        	#else
	        	kernel[i] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	#endif
		}
	}
	else
	{	
		// load data TODO
		/*get_double_hexadecimal_values(input_file_A, A,size_A);
		get_double_hexadecimal_values(input_file_B, B,size_B);
		
		// iniciate C matrix
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	        	h_C[i*size+j] = 0;
	        	d_C[i*size+j] = 0;
	        	
	    	}
		}*/
	}
	// print input
	if (arguments_parameters->print_input)
	{
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	    		#ifdef INT
	    		printf("%d ",A[i*arguments_parameters->size+j]);
	        	#else
	        	printf("%f ",A[i*arguments_parameters->size+j]);
	        	#endif
	    	}
	    	printf("\n");
		}
		printf("\n\n");
		for (int i=0; i < size_k; ++i)
		{
			#ifdef INT
			printf("%d ",kernel[i]);
	        #else
	        printf("%f ",kernel[i]);
	        #endif
		}
		printf("\n\n");

	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////

	// base object init
	GraficObject *conv_bench = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(conv_bench, 0,arguments_parameters->gpu, device);
	if (!arguments_parameters->csv_format_timestamp && !arguments_parameters->csv_format && !arguments_parameters->mute_messages ){
		printf("Using device: %s\n", device);
	}
	
	// init memory
	device_memory_init(conv_bench, arguments_parameters->size * arguments_parameters->size, arguments_parameters->size * arguments_parameters->size, size_k);
	// copy memory to device
	copy_memory_to_device(conv_bench, A, kernel, arguments_parameters->size * arguments_parameters->size, size_k);
	// execute kernel
	execute_kernel(conv_bench, arguments_parameters->size, arguments_parameters->size, arguments_parameters->size, arguments_parameters->kernel_size);
	// copy memory to host
	copy_memory_to_host(conv_bench, d_B, size_matrix);

	// get time
	if (arguments_parameters->print_timing || arguments_parameters->csv_format || arguments_parameters->csv_format_timestamp)
	{
		get_elapsed_time(conv_bench, arguments_parameters->csv_format, arguments_parameters->csv_format_timestamp, get_timestamp());
	}
	if (arguments_parameters->print_output)
	{
		#ifdef INT
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	    		printf("%d ", d_B[i*arguments_parameters->size+j]);
	        	
	    	}
    		printf("\n");
		}
		#else
		for (int i=0; i<arguments_parameters->size; i++){
	    	for (int j=0; j<arguments_parameters->size; j++){
	    		printf("%f ", d_B[i*arguments_parameters->size+j]);
	        	
	    	}
    		printf("\n");
		}
		#endif

		
	}
	


	if (arguments_parameters->verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		matrix_convolution(A,kernel,h_B,arguments_parameters->size,arguments_parameters->kernel_size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (arguments_parameters->print_timing)
		{
			printf("CPU Time %lu milliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (arguments_parameters->print_output)
		{
		#ifdef INT
			for (int i=0; i<arguments_parameters->size; i++){
		    	for (int j=0; j<arguments_parameters->size; j++){
		    		printf("%d ", h_B[i*arguments_parameters->size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#else
			for (int i=0; i<arguments_parameters->size; i++){
		    	for (int j=0; j<arguments_parameters->size; j++){
		    		printf("%f ", h_B[i*arguments_parameters->size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#endif
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
	clean(conv_bench);
	// free object memory 
	free(arguments_parameters);
	free(conv_bench);
	free(A);
	free(kernel);
	free(h_B);
	free(d_B);
return 0;
}


// Arguments part

void print_usage(const char * appName)
{
	printf("Usage: %s -s Size -k [-v] [-e] [-o] [-t] [-d] [-i input_file_A_MATRIX input_file_B_MATRIX] \n", appName);
	printf(" -s Size : set size of x and y of matrices A and B with Size \n");
	printf(" -k: size of the kernel\n");
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verification of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -C: prints the timing in csv format with timestamp\n");
	printf(" -q: prints input values\n");
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
			case 'q' : arguments_parameters->print_input = true;break;
			case 'd' : args +=1; arguments_parameters->gpu = atoi(argv[args]);break;
			case 'f' : arguments_parameters->mute_messages = true;break;
			// specific
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file_A,argv[args]);
					   args +=1;
					   strcpy(arguments_parameters->input_file_B,argv[args]);
					   break;
			case 's' : args +=1; arguments_parameters->size = atoi(argv[args]);break;
			case 'k' : args +=1; arguments_parameters->kernel_size = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( arguments_parameters->size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if ( arguments_parameters->kernel_size <= 0){
		printf("-k need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (arguments_parameters->mute_messages){
		arguments_parameters->csv_format = false;
	}
	return OK_ARGUMENTS;
}