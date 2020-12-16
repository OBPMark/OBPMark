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

int arguments_handler(int argc, char ** argv,unsigned int *size, unsigned int *kernel_size,unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format,bool *print_input, bool *validation_timing, bool *mute_messages,char *input_file_A, char *input_file_B);

int main(int argc, char *argv[]){
	// random init
	srand (21121993);
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	unsigned int size = 0, gpu = 0, kernel_size;
	bool verification  = false, export_results = false, print_output = false, print_timing = false, export_results_gpu = false, csv_format = false, print_input = false, validation_timing = false, mute_messages = false;
	char input_file_A[100] = "";
	char input_file_B[100] = "";

	int resolution = arguments_handler(argc,argv, &size, &kernel_size,&gpu, &verification, &export_results, &export_results_gpu,&print_output, &print_timing, &csv_format, &print_input, &validation_timing, &mute_messages,input_file_A, input_file_B);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// VARIABLES 
	///////////////////////////////////////////////////////////////////////////////////////////////
	// linearizable versions of matrix
	unsigned int size_matrix =size * size;
	// A input matrix
	unsigned int size_A = size * size;
    unsigned int mem_size_A = sizeof(bench_t) * size_A;
	bench_t* A = (bench_t*) malloc(mem_size_A);
	// B input matrix
	unsigned int size_B = size * size;
    unsigned int mem_size_B = sizeof(bench_t) * size_B;
	bench_t* h_B = (bench_t*) malloc(mem_size_B);
	bench_t* d_B = (bench_t*) malloc(mem_size_B);
	// kernel matrix
	unsigned int size_k = kernel_size * kernel_size ;
    unsigned int mem_size_k = sizeof(bench_t) * size_k;
	bench_t* kernel = (bench_t*) malloc(mem_size_k);
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
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	    		#ifdef INT
	        	A[i*size+j] = rand() % (NUMBER_BASE * 100);

	        	#else
	        	A[i*size+j] = (bench_t)rand()/(bench_t)(RAND_MAX/NUMBER_BASE);
	        	#endif
	    	}
		}
	// iniciate B matrix 
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	        	h_B[i*size+j] = 0;
	        	d_B[i*size+j] = 0;
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
	if (print_input)
	{
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	    		#ifdef INT
	    		printf("%d ",A[i*size+j]);
	        	#else
	        	printf("%f ",A[i*size+j]);
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
	// CODE FOR ONLY TIMING  OF THE VALIDATION
	///////////////////////////////////////////////////////////////////////////////////////////////
	if(validation_timing){
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		matrix_convolution(A,kernel,h_B,size,kernel_size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (!mute_messages){
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		exit(0);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// CODE BENCKMARK
	///////////////////////////////////////////////////////////////////////////////////////////////

	// base object init
	GraficObject *conv_benck = (GraficObject *)malloc(sizeof(GraficObject));
	// init devices
	char device[100] = "";
	init(conv_benck, 0,gpu, device);
	if (!csv_format && !mute_messages ){
		printf("Using device: %s\n", device);
	}
	
	// init memory
	device_memory_init(conv_benck, size * size, size * size, size_k);
	// copy memory to device
	copy_memory_to_device(conv_benck, A, kernel, size * size, size_k);
	// execute kernel
	execute_kernel(conv_benck, size, size, size, kernel_size);
	// copy memory to host
	copy_memory_to_host(conv_benck, d_B, size_matrix);

	// get time
	if (print_timing || csv_format)
	{
		get_elapsed_time(conv_benck, csv_format);
	}
	if (print_output)
	{
		#ifdef INT
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	    		printf("%d ", d_B[i*size+j]);
	        	
	    	}
    		printf("\n");
		}
		#else
		for (int i=0; i<size; i++){
	    	for (int j=0; j<size; j++){
	    		printf("%f ", d_B[i*size+j]);
	        	
	    	}
    		printf("\n");
		}
		#endif

		
	}
	


	if (verification)
	{
		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		matrix_convolution(A,kernel,h_B,size,kernel_size);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		if (print_timing)
		{
			printf("CPU Time %lu miliseconds\n", (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000);
		}
		if (print_output)
		{
		#ifdef INT
			for (int i=0; i<size; i++){
		    	for (int j=0; j<size; j++){
		    		printf("%d ", h_B[i*size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#else
			for (int i=0; i<size; i++){
		    	for (int j=0; j<size; j++){
		    		printf("%f ", h_B[i*size+j]);
		        	
		    	}
	    		printf("\n");
			}
		#endif
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
	clean(conv_benck);
	// free object memory 
	free(conv_benck);
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
	printf(" -e: exports the results of the output and the verification in hexadecimal format (this enables the verificaction of the results) \n");
	printf(" -v: verify the output of the gpu program with the cpu output \n");
	printf(" -g: exports the results of the output \n");
	printf(" -o: prints the results\n");
	printf(" -t: prints the timing\n");
	printf(" -c: prints the timing in csv format\n");
	printf(" -q: prints input values\n");
	printf(" -i: pass input data and the result and compares\n");
	printf(" -d: selects GPU\n");
	printf(" -x: prints the timing of the validation. Only the sequential time of the application will be displayed\n");
	printf(" -f: mutes all print\n");
	printf(" -h: print help information\n");
}


int arguments_handler(int argc, char ** argv,unsigned int *size, unsigned int *kernel_size ,unsigned int *gpu,bool *verification, bool *export_results, bool *export_results_gpu,  bool *print_output, bool *print_timing, bool *csv_format,bool *print_input, bool *validation_timing, bool *mute_messages, char *input_file_A, char *input_file_B){
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
			case 'q' : *print_input = true;break;
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
			case 'k' : args +=1; *kernel_size = atoi(argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if ( *size <= 0){
		printf("-s need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if ( *kernel_size <= 0){
		printf("-k need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (*mute_messages){
		*csv_format = false;
	}
	return OK_ARGUMENTS;
}