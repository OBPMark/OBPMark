#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string.h>

#ifndef CPU_LIB_H
#define CPU_LIB_H


#ifdef FLOAT
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

void fft_function(bench_t* data,int64_t nn);
bool compare_vectors(const bench_t* host,const bench_t* device, const int64_t size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);


#endif