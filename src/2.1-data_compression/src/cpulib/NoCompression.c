#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "NoCompression.h"



struct FCompressedData NoCompression(unsigned long int* Samples)
{
    // Output size sanitization
    const unsigned int CompressedSize = non_compressed_size;

    unsigned long int PackedArray[J_BlockSize] = { 0 };
    
    // Rearranging uncompressed block
    for(int i = 0; i < J_BlockSize; ++i)
    {
        for(int bit = 0; bit < n_bits; ++bit)
        {
            if((Samples[(i*32 + bit)/32] & (1 << ((i*32 + bit)%32) )) != 0)
            {
                PackedArray[(i*n_bits+bit)/32] |= 1 << ((i*n_bits+bit)%32);
            }
        }
        
    }

    PRINT_NC_COMPRESSED_ARRAY(PackedArray);

    struct FCompressedData CompressedData;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned long int*) malloc (sizeof( unsigned long int ) * J_BlockSize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));
    CompressedData.CompressionIdentifier = NO_COMPRESSION_ID;


    NC_PRINT(("No Compression (Size: %d bits): OK.\n", CompressedSize));

    return CompressedData;

}