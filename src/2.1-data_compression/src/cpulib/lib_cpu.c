#include <time.h>
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
    unsigned int acc_size = 0;

    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_app);

    // Repeating the operations n times
    for(int step = 0; step < STEPS; ++step)
    {
        // Output initialization Block
        unsigned int OutputPreprocessedValue[device_object->TotalSamples];
        
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
            printf("##########################\n");
			printf("block ID %d\n", block);
            struct FCompressedData it = AdaptativeEntropyEncoder(device_object, OutputPreprocessedValue + (J_BlockSize*block), ZBProcessed[block]);
            acc_size += it.size;
        }

        // The previous loop compresses each block selecting the best compression algorithm per block (decoder).
        // This benckmark serves as a sample compression protocol that encompasses the most data intensive operation of the CCSDS121 standard
        // This code can be extended with the following specification: https://public.ccsds.org/Pubs/121x0b2ec1.pdf
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_app);

    // Debug loop - uncomment if needed
    
    /*for(int i = acc_size - 1; i >= 0; --i)
    {
        printf("%d" ,(device_object->OutputDataBlock[i/32] & (1 << (i%32) )) != 0);
    }
    printf("\n");*/
    
   // Store loop - uncomment if needed stores the output data in a binary file
    
    FILE *fp;
    fp = fopen("output.bin", "wb");
    // acc_size is the number of bits in the output file. So we need to get the number of elements in the array that use 2 bytes
    unsigned int number_of_elements = device_object->OutputDataBlock->num_total_bytes + 1; // add 1 to account for last remaining byte
    printf("Number of elements: %d\n", number_of_elements);
    fwrite(device_object->OutputDataBlock->OutputBitStream, 1, number_of_elements, fp); 
    fclose(fp);

}


void get_elapsed_time(struct DataObject *device_object, bool csv_format)
{
    long unsigned int application_miliseconds = (device_object->end_app.tv_sec - device_object->start_app.tv_sec) * 1000 + (device_object->end_app.tv_nsec - device_object->start_app.tv_nsec) / 1000000;

    if (csv_format)
    {
        printf("%d;%lu;%d;\n", 0, application_miliseconds,0);
    }
    else
    {
        printf("Elapsed time Host->Device: %d miliseconds\n", 0);
        printf("Elapsed time application: %lu miliseconds\n", application_miliseconds);
        printf("Elapsed time Device->Host: %d miliseconds\n", 0);
    }
}


void clean(struct DataObject *device_object)
{
    free(device_object->InputDataBlock);
    free(device_object->OutputDataBlock->OutputBitStream);
    free(device_object->OutputDataBlock);
    free(device_object);
}