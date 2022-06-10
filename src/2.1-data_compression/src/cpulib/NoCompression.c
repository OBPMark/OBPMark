#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "BitOutputUtils.h"
#include "NoCompression.h"



struct FCompressedData NoCompression(unsigned int* Samples)
{
    // Output size sanitization
    struct FCompressedData CompressedData;
    CompressedData.CompressionIdentifierInternal = NO_COMPRESSION_ID;
    CompressedData.size = non_compressed_size;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    memcpy(CompressedData.data, Samples, sizeof(unsigned int)* J_BlockSize);

    if (n_bits < 3)
    {
        CompressedData.CompressionIdentifier = 0x1; //1
    }
    else if (n_bits < 5)
    {
        CompressedData.CompressionIdentifier = 0x3; //11
    }
    else if (n_bits <= 8)
    {
        CompressedData.CompressionIdentifier = 0x7; //111 
    }
    else if (n_bits <= 16)
    {
        CompressedData.CompressionIdentifier = 0xF; //1111 
    }
    else /*if (n_bits <= 32)*/
    {
        CompressedData.CompressionIdentifier = 0x1F; //11111 
    }

    NC_PRINT(("No Compression (Size: %d bits): OK.\n", CompressedSize));

    return CompressedData;
}

void NoCompressionWriter(struct DataObject* device_object, struct FCompressedData* BestCompression)
{
    for(int i = 0; i < J_BlockSize; ++i)
    {
        writeWord(device_object->OutputDataBlock,  BestCompression->data[i], n_bits);
    }
}