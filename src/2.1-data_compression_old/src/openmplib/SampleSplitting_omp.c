#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SampleSplitting.h"
#include "FundamentalSequence.h"


unsigned int GetSizeSampleSplitting(unsigned long int* Samples, unsigned int k)
{
    unsigned int size = 0;
    for(unsigned int i = 0; i < J_BlockSize; ++i)
    {
        size += (k + (Samples[i] >> k) + 1);
    }
    return size;
}


struct FCompressedData SampleSplitting(unsigned long int* Samples, unsigned int k)
{
    struct FCompressedData CompressedData;
    CompressedData.size = J_BlockSize * 32;
    CompressedData.data = NULL;
    CompressedData.CompressionIdentifier = SAMPLE_SPLITTING_ID + k;

    // k sanitization
    if(k >= n_bits - 3)
    {
        // fixmevori: study use case 32 bits per array position, however n_bits is lower.
        SS_PRINT(("Sample Splitting (k >= n_bits - 3): The data stream won't be compressed.\n"));
        return CompressedData;
    }
    else if(k == 0)
    {
        SS_PRINT(("Sample Splitting (k == 0): Performing Fundamental Sequence.\n"));
        return FundamentalSequence(Samples);
    }

    // Output size sanitization
    const unsigned int CompressedSize = GetSizeSampleSplitting(Samples, k);
    if(CompressedSize > J_BlockSize * 32)
    {
        SS_PRINT(("Sample Splitting k = %d (Compressed size: %d bits > %d bits): Error.\n", k, CompressedSize, J_BlockSize * 32));
        return CompressedData;
    }


    unsigned long int PackedArray[J_BlockSize] = { 0 };  
    unsigned int FSSample = 0; 
    unsigned int LeastSignificativeBits = 0;
    unsigned int offset = 0;
    
    for(unsigned int i = 0; i < J_BlockSize; ++i)
    {
        FSSample = Samples[i] >> k;
        LeastSignificativeBits = Samples[i] & ((1UL << k) - 1);
        // Annotate least significative bits
        PackedArray[offset/32] |= LeastSignificativeBits << (offset%32);
        // Annotate Fundamental Sequence
        PackedArray[(offset+k)/32] |= 1 << ((offset+k)%32);
        // Iterating through the array        
        offset += (k + FSSample + 1);
    }

    PRINT_SS_COMPRESSED_ARRAY(PackedArray);

    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned long int*) malloc (sizeof( unsigned long int ) * J_BlockSize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));


    SS_PRINT(("Sample Splitting k = %d (Size: %d bits): OK.\n", k, CompressedSize));

    return CompressedData;

}