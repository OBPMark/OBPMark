#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SecondExtension.h"


unsigned int GetSizeSecondExtension(unsigned long int* HalvedSamples)
{
    unsigned int size = 0;
    for(int i = 0; i < HalfBlockSize; ++i)
    {
        size += HalvedSamples[i] + 1;
    }
    return size;
}


struct FCompressedData SecondExtension(unsigned long int* Samples)
{
    unsigned long int HalvedSamples[HalfBlockSize] = { 0 };
    
    // Halving the data using the SE Option algorithm. See: https://public.ccsds.org/Pubs/121x0b2ec1.pdf
    for(unsigned int i = 0; i < HalfBlockSize; ++i)
    {
        // fixmevori: Non-accumulative consecutive members? 
        HalvedSamples[i] = (((Samples[2*i] + Samples[2*i + 1]) * (Samples[2*i] + Samples[2*i + 1] + 1)) / 2) + Samples[2*i + 1];
        PRINT_HALVED_ARRAY_ELEMENT(i, HalvedSamples[i]);
    }

    struct FCompressedData CompressedData;
    CompressedData.size = J_BlockSize * 32;
    CompressedData.data = NULL;
    CompressedData.CompressionIdentifier = SECOND_EXTENSION_ID;


    // Checks if the compressed size is minor than the uncompressed size 
    const unsigned int CompressedSize = GetSizeSecondExtension(HalvedSamples);
    if(CompressedSize > J_BlockSize * 32)
    {
        SE_PRINT(("Second Extension (Compressed size: %d bits > %d bits): Error.\n", CompressedSize, J_BlockSize * 32));
        return CompressedData;
    }

    // Resulting size is in bounds, we annotate the output PackedArray using the Fundamental Sequence algorithm
    unsigned long int PackedArray[J_BlockSize] = { 0 };    
    unsigned int sample = 0; 
    for(unsigned int i = 0; i < HalfBlockSize; ++i)
    {
        PackedArray[sample/32] |= 1 << (sample%32);
        sample += HalvedSamples[i] + 1;
    }
    
    PRINT_SE_COMPRESSED_ARRAY(PackedArray);

    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned long int*) malloc (sizeof( unsigned long int ) * J_BlockSize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));

    SE_PRINT(("Second Extension (Size: %d bits): OK.\n", CompressedSize));
    
    return CompressedData;
}