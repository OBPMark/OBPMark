/**
 * \file benchmark.h
 * \brief Benchmark #2.2 top-level header
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
#define LEVELS_DWT     3
#define BLOCKSIZEIMAGE 8
#define GAGGLE_SIZE    16
#define UNCODED_VALUE 0xFF

/* Input Defines defines */
#define MINIMUNWSIZE		17
#define MINIMUNHSIZE		17
#define MINSEGMENTSIZE      16
#define DEFAULTSEGMENTSIZE  1024
#define MAXSEGMENTSIZE      1048576


/* Bits defines */
#define MAXIMUNBITSIZEINTEGER		25
#define MAXIMUNBITSIZEFLOAT		    28




/* Low pass filter defines */
#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9

static const float lowpass_filter_cpu[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const float highpass_filter_cpu[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};

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


#endif // BENCHMARK_H_
