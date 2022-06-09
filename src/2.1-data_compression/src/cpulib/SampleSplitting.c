#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SampleSplitting.h"
#include "FundamentalSequence.h"


unsigned int GetSizeSampleSplitting(unsigned int* Samples, unsigned int k)
{
    unsigned int size = 0;
    for(unsigned int i = 0; i < J_BlockSize; ++i)
    {
        size += (k + (Samples[i] >> k) + 1);
    }
    return size;
}


struct FCompressedData SampleSplitting(unsigned int* Samples, unsigned int k)
{
    struct FCompressedData CompressedData;
    CompressedData.size = J_BlockSize * 32;
    CompressedData.data = NULL;
    
    // k sanitization
    if(k >= n_bits - 2)
    {
        // fixmevori: study use case 32 bits per array position, however n_bits is lower.
        SS_PRINT(("Sample Splitting (k >= n_bits - 3): The data stream won't be compressed.\n"));
        CompressedData.CompressionIdentifier = SAMPLE_SPLITTING_ID + k;
        return CompressedData;
    }
    else if(k == 0)
    {
        SS_PRINT(("Sample Splitting (k == 0): Performing Fundamental Sequence.\n"));
        CompressedData.CompressionIdentifier = 1;
        return FundamentalSequence(Samples);
    }

    // Output size sanitization
    const unsigned int CompressedSize = GetSizeSampleSplitting(Samples, k);
    if(CompressedSize > J_BlockSize * 32)
    {
        SS_PRINT(("Sample Splitting k = %d (Compressed size: %d bits > %d bits): Error.\n", k, CompressedSize, J_BlockSize * 32));
        CompressedData.CompressionIdentifier = SAMPLE_SPLITTING_ID + k;
        return CompressedData;
    }

    CompressedData.CompressionIdentifier = k + 1;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    CompressedData.CompressionIdentifierInternal = SAMPLE_SPLITTING_ID + k;
    memcpy(CompressedData.data, Samples, sizeof( unsigned int ) * J_BlockSize);

    SS_PRINT(("Sample Splitting k = %d (Size: %d bits): OK.\n", k, CompressedSize));

    return CompressedData;

}