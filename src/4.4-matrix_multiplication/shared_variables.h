#ifndef SHARED_LIB_H
#define SHARED_LIB_H
#include "../../float_16_lib/include/half.hpp"
#ifdef OPENCL
// OpenCL lib

#else
// CUDA lib
#endif

#ifdef INT
typedef int bench_t;
static const std::string type_kernel = "typedef int bench_t;\n";
#elif FLOAT
typedef float bench_t;
static const std::string type_kernel = "typedef float bench_t;\n";
#elif HALF
// HALF is 
#else
typedef double bench_t;
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
#endif

#endif