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
//bool compare_vectors_int(const int* host,const int* device,const int size);
//bool compare_vectors(const float* host,const float* device, const int size);
bool compare_vectors(const bench_t* host,const bench_t* device, const int size);
void print_double_hexadecimal_values(const char* filename, bench_t* float_vector,  unsigned int size);
void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size);
void set_values_file(char *input_file, double *out_C, unsigned int N);
void get_values_file (char *input_file, bench_t *in_A, bench_t *in_B);


#endif