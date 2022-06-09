
#ifdef OPENCL
#include <iostream>
#include <string>
#include <cstring>
#include <CL/cl.hpp>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Config.h"
#include "lib_functions.h"


void ConfigErrorControl() 
{
    CONFIG_PRINT("Ensuring Config.h: ");
    ensure(J_BlockSize == 8 || J_BlockSize == 16 || J_BlockSize == 32 || J_BlockSize == 64 );
    ensure(n_bits > 0 && n_bits <= 32);
    ensure(r_samplesInterval > 0 && r_samplesInterval <= 4096);
    CONFIG_PRINT("OK.\n");
}


void read_binary_file(char *filename, unsigned int * data)
{
    FILE *file = fopen(filename, "rb");
    if(file == NULL)
    {
        printf("error: failed to open file: %s\n", filename);
        exit(1);
    }
    // read each of the elements in the file
    for (int i = 0; i < J_BlockSize * r_samplesInterval; i++)
    {
        fread(&data[i], sizeof(short int), 1, file);
    }
    fclose(file);
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
    ccsds_data->InputDataBlock = ( unsigned int *)malloc(sizeof( unsigned int ) * TotalSamplesStep);
    
    // Memory initialization - rnd biased
    if (RANDOM_DATA_GENERATION)
    {
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
    }
    else
    {
        // Reading the binary file
        read_binary_file("data.dat", ccsds_data->InputDataBlock);
    }

    // Output allocation
    ccsds_data->OutputDataBlock = (struct OutputBitStream *)malloc (sizeof(struct OutputBitStream));
    // init the output stream
    ccsds_data->OutputDataBlock->OutputBitStream = calloc(TotalSamplesStep*4, sizeof(unsigned char));
    ccsds_data->OutputDataBlock->num_bits = 0;
    ccsds_data->OutputDataBlock->num_total_bytes = 0;
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
