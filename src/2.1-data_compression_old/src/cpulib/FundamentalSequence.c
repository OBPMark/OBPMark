#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "BitOutputUtils.h"
#include "FundamentalSequence.h"


unsigned int GetSizeFundamentalSequence(unsigned int* Samples)
{
    unsigned int size = 0;
    for(int i = 0; i < J_BlockSize; ++i)
    {
        size += Samples[i] + 1;
    }
    return size;
}


struct FCompressedData FundamentalSequence(unsigned int* Samples)
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

    CompressedData.size = CompressedSize;
    CompressedData.CompressionIdentifierInternal = FUNDAMENTAL_SEQUENCE_ID;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    memcpy(CompressedData.data, Samples, sizeof( unsigned int ) * J_BlockSize);

    FS_PRINT(("Fundamental Sequence (Size: %d bits): OK.\n", CompressedSize));
    
    return CompressedData;
}


void FundamentalSequenceWriter(struct DataObject* device_object, struct FCompressedData* BestCompression)
{
    for(int i = 0; i < J_BlockSize; ++i)
    {
        writeValue(device_object->OutputDataBlock, 0 , BestCompression->data[i]);
        writeValue(device_object->OutputDataBlock, 1, 1);
    }
}