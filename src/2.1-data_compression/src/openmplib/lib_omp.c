#include <time.h>
#include <omp.h>
#include <string.h>
#include "lib_functions.h"
#include "Config.h"
#include "AdaptativeEntropyEncoder.h"
#include "Preprocessor.h"


void init(struct DataObject *device_object, int platform ,int device, char* device_name)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
    strcpy(device_name,"Generic device");
}


bool device_memory_init(struct DataObject *device_object)
{
    return true;
}


void execute_benchmark(struct DataObject *device_object)
{
    struct FCompressedData data[STEPS*r_samplesInterval];

    // Start compute timer
	const double start_wtime = omp_get_wtime();

    // Repeating the operations n times
    #pragma omp parallel for
    for(int step = 0; step < STEPS; ++step)
    {
        // Output initialization Block
        unsigned long int OutputPreprocessedValue[device_object->TotalSamples];
        
        // If CheckZeroBlock is false after preprocessing a block, it means that the block contains non zero values
        bool AllZerosInBlock = true;
        unsigned int ZeroCounterPos = 0;
        
        // This array will keep track of the number of consecutive zero blocks storing the position of the last 0 block and the number of zeros
        struct ZeroBlockCounter ZeroCounter[r_samplesInterval];
        for(int i = 0; i < r_samplesInterval; ++i) { ZeroCounter[i].counter = 0; ZeroCounter[i].position = -1; }

        // Pre-processing the input data values, precalculating ZeroBlock offsets
        for(unsigned int block = 0; block < r_samplesInterval; ++block)
        {
            // Preprocessing the samples 
            for(unsigned int i = 0; i < J_BlockSize; ++i)
            {
                #ifdef PREPROCESSOR_ACTIVE
                OutputPreprocessedValue[i + (block * J_BlockSize)] = Preprocessor(device_object->InputDataBlock[(i + (block * J_BlockSize)) + (step * r_samplesInterval * J_BlockSize)]);
                #else
                OutputPreprocessedValue[i + (block * J_BlockSize)] = device_object->InputDataBlock[(i + (block * J_BlockSize)) + (step * r_samplesInterval * J_BlockSize)];
                #endif
                // printf("Value %d, block %d: %ld\n", i, block, OutputPreprocessedValue[(i + (block * J_BlockSize))]);
                if (OutputPreprocessedValue[i + (block * J_BlockSize)] != 0) AllZerosInBlock = false;
            }


            // Zero Block post processing
            if( AllZerosInBlock == true )
            {
                // Increasing the zero count in the record, we found another all-zero block
                ZeroCounter[ZeroCounterPos].counter++;
                ZeroCounter[ZeroCounterPos].position = block;

            }
            else if(ZeroCounter[ZeroCounterPos].position != -1)
            {
                // Increasing ZeroCounterPos only if we found a non zero numbers
                ZeroCounterPos++;
                AllZerosInBlock = true;
            }
        }
        
        // ZeroBlock processed array per position
        struct ZeroBlockProcessed ZBProcessed[r_samplesInterval] = { {-1} };
        for(int i = 0; i < r_samplesInterval; ++i) { ZBProcessed[i].NumberOfZeros = -1; }

        
        // Processing the ZeroBlock Arrays: Adding the number of 0's to be written per block
        for(unsigned int i = 0; i < ZeroCounterPos; ++i)
        {        
            for(unsigned int pos = 0; pos < ZeroCounter[i].counter; ++pos)
            {
                // Calculating ROS (Remainder of a segment)
                const short int bSkip5th = !((ZeroCounter[i].counter < 9 && (ZeroCounter[i].position - pos) >= 4) || (ZeroCounter[i].counter >= 9 && (ZeroCounter[i].position - pos) >= 5));
                const unsigned int z_number = ZeroCounter[i].counter - pos - bSkip5th;
                ZBProcessed[ZeroCounter[i].position - pos].NumberOfZeros = z_number;
            }
        }

        // Compression formatting: [header n_bits] [compressed blocks]
        // The header simply consists on the selected number of bits, so we can decompress the no-compression method.
        PRINT_HEADER(n_bits);

        // Compressing each block
        // The formatting for each compressed block is further specificed in the AdaptativeEntropyEncoder.c file.
        for(unsigned int block = 0; block < r_samplesInterval; ++block)
        {  
            data[block+(r_samplesInterval*step)] = AdaptativeEntropyEncoder(OutputPreprocessedValue + (J_BlockSize*block), ZBProcessed[block]);
            
        }

        // The previous loop compresses each block selecting the best compression algorithm per block (decoder).
        // This benckmark serves as a sample compression protocol that encompasses the most data intensive operation of the CCSDS121 standard
        // This code can be extended with the following specification: https://public.ccsds.org/Pubs/121x0b2ec1.pdf
    }

    // In OpenMP we merge the data in a separate step so the outer loop is parallelisable
    unsigned int acc_size = 0;
    for(unsigned int i = 0; i < STEPS*r_samplesInterval; ++i)
    {
        if(i % r_samplesInterval == 0)
        {
            for(unsigned char bit = 0; bit < 6; ++bit)
            {
                if((n_bits & (1 << (bit%8) )) != 0)
                {
                    device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
                }
            }
            acc_size += 6;
        }
        
        struct FCompressedData it = data[i];
        // Rearranging compressed blocks
        for(int bit = 0; bit < it.size; ++bit)
        {
            if((it.data[bit/32] & (1 << (bit%32) )) != 0)
            {
                device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
            }
        }
        acc_size += it.size;
    }

    // End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;

    // Debug loop - uncomment if needed
    /*
    for(int i = acc_size - 1; i >= 0; --i)
    {
        printf("%d" ,(device_object->OutputDataBlock[i/32] & (1 << (i%32) )) != 0);
    }
    printf("\n");
    */
}


void get_elapsed_time(struct DataObject *device_object, bool csv_format)
{
    if (csv_format)
    {
        printf("%d;%f;%d;\n", 0, device_object->elapsed_time * 1000.f, 0);
    }
    else
    {
        printf("Elapsed time Host->Device: %d miliseconds\n", 0);
        printf("Elapsed time application: %f miliseconds\n", device_object->elapsed_time * 1000.f);
        printf("Elapsed time Device->Host: %d miliseconds\n", 0);
    } 
}


void clean(struct DataObject *device_object)
{
    free(device_object->InputDataBlock);
    free(device_object->OutputDataBlock);
    free(device_object);
}   