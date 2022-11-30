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

#define MINIMUMBITSIZE		0
#define MINIMUMRSAMPLES		0
#define MINIMUMSAMPLINGINTERVAL 4


#define MAXIMUNBITSIZE      32

#define JBLOCKSIZE1          8
#define JBLOCKSIZE2          16
#define JBLOCKSIZE3          32
#define JBLOCKSIZE4          64

/* Internal Identifiers */
#define ZERO_BLOCK_ID 0
#define FUNDAMENTAL_SEQUENCE_ID 1
#define SECOND_EXTENSION_ID 2
#define SAMPLE_SPLITTING_ID 3
#define NO_COMPRESSION_ID 32

#define MAX_NUMBER_OF_BLOCKS 4096

/* Device defines */
#ifdef CUDA
#define NUMBER_STREAMS 2
#define MAXSIZE_NBITS 32
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE 	256 
#define BLOCK_SIZE 		16

#elif OPENCL
#define NUMBER_STREAMS 2
#define MAXSIZE_NBITS 32
#define DEVICESELECTED		0
#define BLOCK_SIZE_PLANE 	256 
#define BLOCK_SIZE 		16

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
