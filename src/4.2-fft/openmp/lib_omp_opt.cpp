#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, int64_t size)
{
   	device_object->d_Br = (bench_t*) malloc ( size * sizeof(bench_t*));
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size)
{
	device_object->d_B = h_B;
}


void execute_kernel(GraficObject *device_object, int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int mode = (unsigned int)log2(size);
	unsigned int position = 0;
	unsigned int j, i, z, l, istep, mmax, n, m = 0;

	#pragma omp parallel for private(j,i,position)
	for(i = 0; i < size; ++i)
	{
		j = i;                                                                                                    
		j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
		j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
		j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
		j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
		j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
		j >>= (32-mode);                                                                                                       
		position = j * 2;                                                                                                       																											
		device_object->d_Br[position] = device_object->d_B[i *2];                                                                                                
		device_object->d_Br[position + 1] = device_object->d_B[i *2 + 1];  
	}

	bench_t wpr, wpi, theta, wi, tempr, tempi, wtemp, wr = 0.f;
    mmax=2;
	n = size << 1;

	bench_t* a = device_object->d_Br;

    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
		#pragma omp task untied
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
				j=i+mmax;
				tempr = wr*a[j-1] - wi*a[j];
				tempi = wr * a[j] + wi*a[j-1];
                a[j-1] = a[i-1] - tempr;
                a[j] = a[i] - tempi;
                a[i-1] += tempr;
                a[i] += tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
    }
	
	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{	     
	memcpy(h_B, &device_object->d_Br[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format)
{
	if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_Br);
}