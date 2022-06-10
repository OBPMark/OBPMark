#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "BitOutputUtils.h"
#include "ZeroBlock.h"

// fimevori: Experimental WIP module: This module requires certain clarifications from the CCSDS committee/experts.


struct FCompressedData ZeroBlock(unsigned int* Samples, unsigned int NumberOfZeros)
{
    // Output size sanitization
    const unsigned int CompressedSize = NumberOfZeros + 1;

    unsigned int PackedArray[J_BlockSize] = { 0 };
    PackedArray[0] = 1;

    struct FCompressedData CompressedData;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));
    CompressedData.CompressionIdentifier = ZERO_BLOCK_ID;
    CompressedData.CompressionIdentifierInternal = ZERO_BLOCK_ID;

    ZB_PRINT(("Zero Block (Size: %d bits): OK.\n", CompressedSize));

    return CompressedData;
}

void ZeroBlockWriter(struct DataObject* device_object, struct FCompressedData* BestCompression)
{
    writeValue(device_object->OutputDataBlock,  0, BestCompression->size - 1);
    writeValue(device_object->OutputDataBlock, 1,1);
}
