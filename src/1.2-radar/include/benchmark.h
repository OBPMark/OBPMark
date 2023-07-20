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
#ifdef FFT_LIB
#include <cufft.h>
#endif
#define IMPL_NAME "CUDA"
#define IMPL_NAME_FILE "cuda"
#define FFTL_NAME "CUFFT"
#elif OPENCL
/* OPENCL version */
#include <CL/cl.hpp>
#include <iostream>
#define IMPL_NAME "OpenCL"
#define IMPL_NAME_FILE "opencl"
#elif OPENMP
/* OPENMP version */
#include <omp.h>
#ifdef FFT_LIB
#include <fftw3.h>
#endif
#define IMPL_NAME "OpenMP"
#define IMPL_NAME_FILE "openmp"
#define FFTL_NAME "FFTW3"
#elif HIP
#include "hip/hip_runtime.h"
#define IMPL_NAME "HIP"
#define IMPL_NAME_FILE "hip"
#define FFTL_NAME "HIPFFT"
#else
#define IMPL_NAME "CPU"
#define IMPL_NAME_FILE "cpu"
#endif
#ifdef FFT_LIB
#define IMPLEMENTATION_NAME IMPL_NAME " + " FFTL_NAME
#define IMPLEMENTATION_NAME_FILE IMPL_NAME_FILE "lib"
#else
#define IMPLEMENTATION_NAME IMPL_NAME
#define IMPLEMENTATION_NAME_FILE IMPL_NAME_FILE
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
#define BLOCK_SIZE 		1024
#define TILE_SIZE 		32

#elif OPENCL
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE	256
#define BLOCK_SIZE		1024
#define TILE_SIZE 		32

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
