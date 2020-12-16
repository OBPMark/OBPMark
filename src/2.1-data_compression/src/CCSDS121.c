
#ifdef OPENCL
#include <iostream>
#include <string>
#include <cstring>
#include <CL/cl.hpp>
#endif

#include <stdio.h>
//#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "Config.h"
//#include "Preprocessor.h"
//#include "AdaptativeEntropyEncoder.h"
#include "lib_functions.h"


void ConfigErrorControl() 
{
    CONFIG_PRINT("Ensuring Config.h: ");
    ensure(J_BlockSize == 8 || J_BlockSize == 16 || J_BlockSize == 32 || J_BlockSize == 64);
    ensure(n_bits > 0 && n_bits <= 32);
    ensure(r_samplesInterval > 0 && r_samplesInterval <= 4096);
    CONFIG_PRINT("OK.\n");
}


int main() 
{
    // Ensures that the config parameters are set correctly
    ConfigErrorControl();

    // Seeding the rand algorithm
    srand(8111995);

    const unsigned int TotalSamples = J_BlockSize * r_samplesInterval; 
    const unsigned int TotalSamplesStep = TotalSamples * STEPS;
    
     // base object init
    struct DataObject *ccsds_data = (struct DataObject *)malloc(sizeof(struct DataObject));
    ccsds_data->InputDataBlock = ( unsigned long int *)malloc(sizeof( unsigned long int ) * TotalSamplesStep);
    
    // Memory initialization - rnd biased
    for(unsigned int i = 0; i < TotalSamplesStep; ++i)
    {
        if(i >= 0 && i < 9*J_BlockSize) 
        {
            // Forcing the 9 first blocks to be all 0's.
            ccsds_data->InputDataBlock[i] = 0;
        }
        else
        {
            ccsds_data->InputDataBlock[i] = rand() % (1UL << n_bits);
        }
    }

    // Output allocation
    ccsds_data->OutputDataBlock = ( unsigned long int *)malloc(sizeof( unsigned long int ) * TotalSamplesStep);
    ccsds_data->TotalSamples = TotalSamples;
    ccsds_data->TotalSamplesStep = TotalSamplesStep;

    // Benchmark interface
    char device[30] = "";
    init(ccsds_data, 0, 0, device);
    device_memory_init(ccsds_data);
    execute_benchmark(ccsds_data);
    get_elapsed_time(ccsds_data, true);
    clean(ccsds_data);

}
