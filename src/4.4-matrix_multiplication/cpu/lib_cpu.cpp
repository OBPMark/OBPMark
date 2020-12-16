#include "lib_cpu.h"

void matrix_multiplication(const bench_t* A, const bench_t* B, bench_t* C,const unsigned int n, const unsigned int m, const unsigned int w ){
	for (unsigned int i = 0; i < n; ++i)
	{
		for (unsigned int j = 0; j < w; ++j)
		{
			for (unsigned int k = 0; k < m; ++k)
			{   
				C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*w+j];
			}
		}
	}

}

bool compare_vectors(const bench_t* host,const bench_t* device, const int size){
	#ifdef INT
	for (int i = 0; i < size; ++i){
		if (host[i] != device[i]){
			printf("Error in element %d is %d but was %d\n", i,device[i], host[i]);
			return false;
		}
	}
	return true;
	#else 
		for (int i = 0; i < size; ++i){
			if (fabs(host[i] - device[i]) > 1e-3){
				printf("Error in element %d is %f but was %f\n", i,device[i], host[i]);
				return false;
			}
		}
		return true;
	#endif
}
void writeDouble(double *_d, FILE* _f){
	double locD;
	int res;

#ifdef BIGENDIAN
	unsigned char *r, *w;
	int i;
	r = (unsigned char*)_d;
	w = (unsigned char*)&locD;
	for(i=0; i<8; i++){
		w[i] = r[7-i];
	}
#else
	locD = *_d;
#endif
	res =fwrite((void*)&locD, sizeof(double), 1, _f);
	if(res != 1){
		printf("writeDouble error (%d)\n", res);
		exit(1);
	}
}

void readDouble(double *_d, FILE* _f){
	double locD;
	int res;
	res = fread(&locD, sizeof(double), 1, _f);
	if(res != 1){
		printf("readDouble error\n");
		exit(1);
	}
#ifdef BIGENDIAN
	unsigned char *r, *w;
	int i;
	r = (unsigned char*)&locD;
	w = (unsigned char*)_d;
	for(i=0; i<8; i++){
		w[i] = r[7-i];
	}
#else
	*_d = locD;
#endif
}


void get_values_file (char *input_file, double *in_A, bench_t *in_B){
	FILE *f;
	double D;
	int N, i;
	f=fopen(input_file, "r+b");
	if(f == NULL)
	{
		printf("Error opening file: %s\n", input_file);
	}
	readDouble(&D, f);
	N = (int)D;
	printf("N = %d\n", N);
	//in_A    = (bench_t*)malloc(N*N*sizeof(bench_t));
	//in_B    = (bench_t*)malloc(N*N*sizeof(bench_t));
	for(i=0; i<N*N; i++){
		readDouble(&D, f);
		in_A[i] = D;
	}
	for(i=0; i<N*N; i++){
		readDouble(&D, f);
		in_B[i] = D;
	}
	fclose(f);
}

void set_values_file(char *input_file, double *out_C, unsigned int N){
	FILE *f;
	f=fopen(input_file, "w");
	double D;
	int i;
	if(f == NULL)
	{
		printf("Error opening file: %s\n", input_file);
	}
	//write size
	D = N;
	writeDouble(&D,f);
	for(i=0; i<N*N; i++){
		writeDouble(&out_C[i], f);
	}
	fclose(f);
}

void print_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size){
	FILE *output_file = fopen(filename, "w");
  	// file created
  	for (unsigned int i = 0; i < size; ++i){
  		binary_float.f = float_vector[i];
		fprintf(output_file, "%02x", binary_float.binary_values.a );
		fprintf(output_file, "%02x", binary_float.binary_values.b );
		fprintf(output_file, "%02x", binary_float.binary_values.c );
		fprintf(output_file, "%02x", binary_float.binary_values.d );
		fprintf(output_file, "%02x", binary_float.binary_values.e );
		fprintf(output_file, "%02x", binary_float.binary_values.f );
		fprintf(output_file, "%02x", binary_float.binary_values.g );
		fprintf(output_file, "%02x", binary_float.binary_values.h );
		fprintf(output_file, "\n"); 
  	}
  	fclose(output_file);	

}

void get_double_hexadecimal_values(const char* filename, bench_t* float_vector, unsigned int size){
	// open file
	FILE *file = fopen(filename, "r");
	// read line by line
	char * line = NULL;
    size_t len = 0;
    

	for (unsigned int i = 0; i < size; ++i){
		getline(&line, &len, file);
		// delete /n
		line[strlen(line)-1] = 0;
		// strip for each char
		char *temp = (char*) malloc(sizeof(char) * 2);
		char *ptr;
    	temp[0] = line[0];
		temp[1] = line[1];
    	binary_float.binary_values.a = (char)strtol(temp, &ptr, 16);
		temp[0] = line[2];
		temp[1] = line[3];
		binary_float.binary_values.b = (char)strtol(temp, &ptr, 16);
		temp[0] = line[4];
		temp[1] = line[5];
		binary_float.binary_values.c = (char)strtol(temp, &ptr, 16);
		temp[0] = line[6];
		temp[1] = line[7];
		binary_float.binary_values.d = (char)strtol(temp, &ptr, 16);
		temp[0] = line[8];
		temp[1] = line[9];
		binary_float.binary_values.e = (char)strtol(temp, &ptr, 16);
		temp[0] = line[10];
		temp[1] = line[11];
		binary_float.binary_values.f = (char)strtol(temp, &ptr, 16);
		temp[0] = line[12];
		temp[1] = line[13];
		binary_float.binary_values.g = (char)strtol(temp, &ptr, 16);
		temp[0] = line[14];
		temp[1] = line[15];
		binary_float.binary_values.h = (char)strtol(temp, &ptr, 16);

		float_vector[i] = binary_float.f;
	}
  	fclose(file);	

}
