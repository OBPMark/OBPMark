#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "NoCompression.h"



struct FCompressedData NoCompression(unsigned int* Samples)
{
    // Output size sanitization
    const unsigned int CompressedSize = non_compressed_size;

    /*unsigned int PackedArray[J_BlockSize];
    for(int i = 0; i < J_BlockSize; ++i)
    {
        PackedArray[i] = 0;
    }*/
    
    // Rearranging uncompressed block
    /*for(int i = 0; i < J_BlockSize; ++i)
    {
        for(int bit = 0; bit < n_bits; ++bit)
        {
            if((Samples[(i*32 + bit)/32] & (1 << ((i*32 + bit)%32) )) != 0)
            {
                // if bit is between 0 and 15 then store in the second half of the array
                if(bit < 16)
                {
                    PackedArray[((i*n_bits + bit))/32] |= 1 << ((i*n_bits + bit + n_bits)%32);
                }
                else
                {
                    PackedArray[((i*n_bits + bit))/32] |= 1 << ((i*n_bits + bit - n_bits)%32);
                }
                
            }
        }
    }*/


    //PRINT_NC_COMPRESSED_ARRAY(PackedArray);

    struct FCompressedData CompressedData;
    CompressedData.CompressionIdentifierInternal = NO_COMPRESSION_ID;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * J_BlockSize);
    memcpy(CompressedData.data, Samples, sizeof(unsigned int)* J_BlockSize);

    
    if (n_bits < 3){
        CompressedData.CompressionIdentifier = 0x1; //1
    }
    else if (n_bits < 5){
        CompressedData.CompressionIdentifier = 0x3; //11
    }
    else if (n_bits > 4 && n_bits <= 8){
        CompressedData.CompressionIdentifier = 0x7; //111 
    }
    else if (n_bits > 8 && n_bits <= 16){
        CompressedData.CompressionIdentifier = 0xF; //1111 
    }
    else if (n_bits > 16 && n_bits <= 32){
        CompressedData.CompressionIdentifier = 0x1F; //11111 
    }
    else {
        CompressedData.CompressionIdentifier = 0x1F; //11111
    }


    NC_PRINT(("No Compression (Size: %d bits): OK.\n", CompressedSize));

    return CompressedData;

}