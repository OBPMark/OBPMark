/**
 * \file benchmark.h
 * \brief Benchmark #1.1 top-level header
 * \author Ivan Rodriquez (BSC)
 */
#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "obpmark.h"

//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

/* Device libraries */
#ifdef CUDA
#include <cuda_runtime.h>
#define IMPLEMENTATION_NAME "CUDA"
#define IMPLEMENTATION_NAME_FILE "cuda"
#elif OPENCL
/* OPENCL version */
#include <CL/cl.hpp>
#include <iostream>
#define IMPLEMENTATION_NAME "OpenCL"
#define IMPLEMENTATION_NAME_FILE "opencl"
#elif OPENMP
/* OPENMP version */
#include <omp.h>
#define IMPLEMENTATION_NAME "OpenMP"
#define IMPLEMENTATION_NAME_FILE "openmp"
#elif HIP
#include "hip/hip_runtime.h"
#define IMPLEMENTATION_NAME "HIP"
#define IMPLEMENTATION_NAME_FILE "hip"
#else
#define IMPLEMENTATION_NAME "CPU"
#define IMPLEMENTATION_NAME_FILE "cpu"
#endif

/* Defines */
/* Print Defines */


/* Frames defines */
#define MINIMUNWSIZE		1024
#define MINIMUNHSIZE		1024
#define MINIMUNFRAMES 		1
#define FRAMEBUFFERSIZE     5

/* Bits defines */
#define MINIMUNBITSIZE		14
#define MAXIMUNBITSIZE		16

/* Device defines */
#ifdef CUDA
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE 	256 
#define BLOCK_SIZE 		16

#elif OPENCL
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE	256
#define BLOCK_SIZE		16

#elif OPENMP
#define DEVICESELECTED		0

#elif HIP
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE 	256 
#define BLOCK_SIZE 		16

#else
#define DEVICESELECTED		0
#endif

/* Bad pixel defines with 900/1000 is a 0.1 % of bad pixel */
#define BADPIXELTHRESHOLD	900
#define MAXNUMBERBADPIXEL	1000

#endif // BENCHMARK_H_
