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


void matrix_convolution(const bench_t* A, bench_t* kernel, bench_t* B,const int size, const int kernel_size){
//loop for the image
	int kernel_rad = kernel_size / 2;
	for (int x = 0; x < size; ++x)
	{
		for (int y = 0; y < size; ++y)
		{	
			bench_t sum = 0;
			//loop over the kernel
			for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
			{
				for(int j = -kernel_rad; j <= kernel_rad; ++j){
					// get value
					bench_t value = 0;
					
					if (i + x < 0 || j + y < 0)
					{
						value = 0;
						//printf("ENTRO %d %d\n", i + x , j + y);
					}
					else if ( i + x > size - 1 || j + y > size - 1)
					{
						value = 0;
						//printf("ENTRO UPPER%d %d\n", i + x , j + y);
					}
					else
					{
						value = A[(x + i)*size+(y + j)];
					}
					//printf("ACHIVED position  %d %d value %f\n", (x + i) , (y + j), value);
					sum += value * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
				}
			}
			
			B[x*size+y ] = sum;
		}
	}

}
void vector_convolution(const bench_t* A, bench_t* kernel, bench_t* B,const int size, const int kernel_size){
	int kernel_radious = kernel_size/2;
	int output_size = size + kernel_size - 1;
	for(int i = 0;i < output_size;++i)
	{
  		
		for (int j = 0; j< kernel_size; ++j){
			 
			if (i +(j - kernel_size + 1) >= 0 && i +(j - kernel_size +1)<  size)
    		{	
    			
    			B[i] += kernel[kernel_size - j - 1] * A[i +(j - kernel_size + 1) ];
    		}
    		else
    		{
    			B[i] += 0;
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
			if (fabs(host[i] - device[i]) > 1e-4){
				printf("Error in element %d is %f but was %f\n", i,device[i], host[i]);
				return false;
			}
		}
		return true;
	#endif
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
