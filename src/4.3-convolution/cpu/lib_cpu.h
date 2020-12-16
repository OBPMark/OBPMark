#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string.h>

#ifndef CPU_LIB_H
#define CPU_LIB_H

#ifdef INT
typedef int bench_t;
#elif FLOAT
typedef float bench_t;
#else 
typedef double bench_t;
#endif

#ifdef BIGENDIAN
// bigendian version
union
	{
		double f;
		struct
		{
			unsigned char a,b,c,d,e,f,g,h;
		}binary_values;
	} binary_float;
#else
// littelendian version
union
	{
		double f;
		struct
		{
			unsigned char h,g,f,e,d,c,b,a;
		}binary_values;
	} binary_float;
#endif

void matrix_multiplication(const bench_t* A, const bench_t* B, bench_t* C,const unsigned int n, const unsigned int m, const unsigned int w );
void matrix_convolution(const bench_t* A, bench_t* kernel, bench_t* B,const int size, const int kernel_size);
//bool compare_vectors_int(const int* host,const int* device,const int size);
//bool compare_vectors(const float* host,const float* device, const int size);
bool compare_vectors(const bench_t* host,const bench_t* device, const int size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);


#endif