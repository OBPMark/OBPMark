/**
 * \file benchmark.h
 * \brief Benchmark #1.2 top-level header
 * \author Marc Sole Bonet (BSC)
 */
#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "obpmark.h"

//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <float.h>
#include <math.h>
#include <complex>

/* Device libraries */
#ifdef CUDA
#include <cuda_runtime.h>
#elif OPENCL
/* OPENCL version */
#include <CL/cl.hpp>
#include <iostream>
#elif OPENMP
/* OPENMP version */
#include <omp.h>
#elif HIP
#include "hip/hip_runtime.h"
#else
#endif

/* Defines */

/* Frames defines */
#define MINIMUNWSIZE		4
#define MINIMUNHSIZE		4

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

const float pi = (float) M_PI;      //PI
const float c = (float) 299792458;  //speed of light


#endif // BENCHMARK_H_
