#include <iostream>
#include <stdio.h>
#include <stdlib.h>


#ifdef FLOAT
#define __ptype "%f"
typedef float bench_t;
static const std::string type_kernel = "typedef float bench_t;\n";
#elif DOUBLE
#define __ptype "%f"
typedef double bench_t;
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
#endif

#ifdef OPENCL
// OpenCL lib
//#include <CL/opencl.h>
#include <CL/cl.hpp>
#elif OPENMP
// OpenMP lib
#include <omp.h>
#else
// CUDA lib
#include <cuda_runtime.h>
#include <cufft.h>
#ifdef FLOAT
typedef cufftComplex bench_cuda_complex;
#else 
typedef cufftDoubleComplex bench_cuda_complex;
#endif
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct GraficObject{
   	#ifdef OPENCL
// OpenCL PART
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device default_device;
	cl::Event *evt_copyB;
	cl::Event *evt_copyBr;
	cl::Event *evt;
	cl::Buffer *d_B;
	cl::Buffer *d_Br;
	#elif OPENMP
	// OpenMP part
	bench_t* d_B;
	bench_t* d_Br;
	#else
	// CUDA PART
	#ifdef LIB
	bench_cuda_complex* d_B;
	#else
	bench_t* d_B;
	bench_t* d_Br;
	#endif
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start;
	cudaEvent_t *stop;
	#endif
	float elapsed_time;
};

void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(GraficObject *device_object, int64_t size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size);
void execute_kernel(GraficObject *device_object, int64_t n);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size);
float get_elapsed_time(GraficObject *device_object, bool csv_format);
void clean(GraficObject *device_object);


#endif