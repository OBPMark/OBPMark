#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "FundamentalSequence.h"


unsigned int GetSizeFundamentalSequence(unsigned long int* Samples)
{
    unsigned int size = 0;
    for(int i = 0; i < J_BlockSize; ++i)
    {
        size += Samples[i] + 1;
    }
    return size;
}


struct FCompressedData FundamentalSequence(unsigned long int* Samples)
{   
    struct FCompressedData CompressedData;
    CompressedData.size = J_BlockSize * 32;
    CompressedData.data = NULL;
    CompressedData.CompressionIdentifier = FUNDAMENTAL_SEQUENCE_ID;

    // Checks if the compressed size is minor than the uncompressed size 
    const unsigned int CompressedSize = GetSizeFundamentalSequence(Samples);
    if(CompressedSize > J_BlockSize * 32)
    {
        FS_PRINT(("Fundamental Sequence (Compressed size: %d bits > %d bits): Error.\n", CompressedSize, J_BlockSize * 32));
        return CompressedData;
    }

    // Resulting size is in bounds, we annotate the output PackedArray using the Fundamental Sequence algorithm
    // See: https://public.ccsds.org/Pubs/121x0b2ec1.pdf
    unsigned long int PackedArray[J_BlockSize] = { 0 };    
    unsigned int sample = 0; 
    for(unsigned int i = 0; i < J_BlockSize; ++i)
    {
        PackedArray[sample/32] |= 1 << (sample%32);
        sample += Samples[i] + 1;
    }
    
    PRINT_FS_COMPRESSED_ARRAY(PackedArray);

    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned long int*) malloc (sizeof( unsigned long int ) * J_BlockSize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));

    FS_PRINT(("Fundamental Sequence (Size: %d bits): OK.\n", CompressedSize));
    
    return CompressedData;
}