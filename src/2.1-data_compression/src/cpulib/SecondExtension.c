#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SecondExtension.h"


unsigned int GetSizeSecondExtension(unsigned int* HalvedSamples)
{
    unsigned int size = 0;
    for(int i = 0; i < HalfBlockSize; ++i)
    {
        size += HalvedSamples[i] + 1;
    }
    return size;
}


struct FCompressedData SecondExtension(unsigned int* Samples)
{
    unsigned int HalvedSamples[HalfBlockSize] = { 0 };
    
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
    CompressedData.CompressionIdentifier = SECOND_EXTENSION_ID - 1;

    // Checks if the compressed size is minor than the uncompressed size 
    const unsigned int CompressedSize = GetSizeSecondExtension(HalvedSamples);
    if(CompressedSize > J_BlockSize * 32)
    {
        SE_PRINT(("Second Extension (Compressed size: %d bits > %d bits): Error.\n", CompressedSize, J_BlockSize * 32));
        return CompressedData;
    }
    
    PRINT_SE_COMPRESSED_ARRAY(HalvedSamples);

    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    CompressedData.CompressionIdentifierInternal = SECOND_EXTENSION_ID;
    memcpy(CompressedData.data, HalvedSamples, sizeof( unsigned int ) * HalfBlockSize);

    SE_PRINT(("Second Extension (Size: %d bits): OK.\n", CompressedSize));
    
    return CompressedData;
}