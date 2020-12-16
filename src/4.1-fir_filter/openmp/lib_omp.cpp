#include "../benchmark_library.h"
#include <cstring>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix)
{
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b)
{
	device_object->d_A = h_A;
	device_object->kernel = kernel;
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	const unsigned int kernel_rad = kernel_size / 2;
	const unsigned int output_size = n + kernel_size - 1;

	#pragma omp parallel for
	for(unsigned int i = 0; i < output_size; ++i)
	{
		for (unsigned int j = 0; j < kernel_size; ++j)
		{		 
			if (i +(j - kernel_size + 1) >= 0 && i +(j - kernel_size +1)<  n)
    		{	
    			device_object->d_B[i] += device_object->kernel[kernel_size - j - 1] * device_object->d_A[i +(j - kernel_size + 1) ];
    		}
		}
	}
	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format)
{
	if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_B);
}