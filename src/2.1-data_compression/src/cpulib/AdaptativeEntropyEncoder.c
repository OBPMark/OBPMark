#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "FundamentalSequence.h"
#include "SampleSplitting.h"
#include "SecondExtension.h"
#include "ZeroBlock.h"
#include "NoCompression.h"
#include "AdaptativeEntropyEncoder.h"
#include "Config.h"
#include "BitOutputUtils.h"

// This min method already ensures memory freeing
struct FCompressedData MIN(struct FCompressedData a, struct FCompressedData b)
{
    if(a.size < b.size)
    {
        if(b.data != NULL) free(b.data);
        return a;
    }
    else
    {
        if(a.data != NULL) free(a.data);
        return b;
    }
}

void AdaptativeEntropyEncoder(struct DataObject* device_object , unsigned int* Samples, struct ZeroBlockProcessed ZeroNum)
{
    struct FCompressedData BestCompression;

    if(ZeroNum.NumberOfZeros == -1)
    {
        const struct FCompressedData size_no_compression = NoCompression(Samples);
        const struct FCompressedData size_se  = SecondExtension(Samples);
        
        BestCompression = MIN(size_no_compression, size_se);
        
        // Sample splitting k = i
        for(int i = 0; i < n_bits; ++i)
        {
            BestCompression = MIN(SampleSplitting(Samples, i), BestCompression);
        }  

    }
    else
    {
       BestCompression = ZeroBlock(Samples, ZeroNum.NumberOfZeros);
    }
    
    AEE_PRINT(("Selected Compression Method: %d with %d size.\n", BestCompression.CompressionIdentifier, BestCompression.size));

    /* Once we select the Compression method to apply, we have to encode the output result 
     In this specific case, we decided to encode the output of the AEE the following way:
     [ DATA: Variable length bit ] [ Compression Technique Identification: n bit field ]
    */
    
    unsigned int compression_technique_identifier_size = 1;
    // define the size of the compression technique identifier base of n_bits size
    if (n_bits < 3){
        compression_technique_identifier_size = 1;
    }
    else if (n_bits < 5)
    {
        compression_technique_identifier_size = 2;
    }
    else if (n_bits <= 8)
    {
        compression_technique_identifier_size = 3;
    }
    else if (n_bits <= 16)
    {
        compression_technique_identifier_size = 4;
    }
    else /*if (n_bits <= 32)*/
    {
        compression_technique_identifier_size = 5;
    }

    // If the selected technique is the zero block or the second extension, the compression_technique_identifier_size is +1
    if(BestCompression.CompressionIdentifierInternal == ZERO_BLOCK_ID || BestCompression.CompressionIdentifierInternal == SECOND_EXTENSION_ID)
    {
        compression_technique_identifier_size += 1;
    }


    
    // The sizeof an unsigned short is 16 bit. We simply can use this conversion method to have a good bit stream for this field.
    const unsigned char CompressionTechniqueIdentifier = BestCompression.CompressionIdentifier;
    
    // This array will return the standarized output for the AEE
    printf("Compression Technique Identifier: %d\n", CompressionTechniqueIdentifier);
    writeWordChar(device_object->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);
    
    
    // write each bit of the compressed data
    if (BestCompression.CompressionIdentifierInternal == ZERO_BLOCK_ID)
    {
        // write the 0
        writeValue(device_object->OutputDataBlock,  0, BestCompression.size - 1);
        // write the last one
        writeValue(device_object->OutputDataBlock, 1,1);
    }
    else if(BestCompression.CompressionIdentifierInternal == NO_COMPRESSION_ID)
    {
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeWord(device_object->OutputDataBlock,  BestCompression.data[i], n_bits);
        }
        
        
    }
    else if(BestCompression.CompressionIdentifierInternal == FUNDAMENTAL_SEQUENCE_ID)
    {
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeValue(device_object->OutputDataBlock, 0 , BestCompression.data[i]);
            writeValue(device_object->OutputDataBlock, 1, 1);
        }
    }
    else if(BestCompression.CompressionIdentifierInternal == SECOND_EXTENSION_ID)
    {
         for(int i = 0; i < HalfBlockSize; ++i)
        {
            writeWord(device_object->OutputDataBlock,  BestCompression.data[i], sizeof(unsigned int) * 8);
        }
    }
    else if(BestCompression.CompressionIdentifierInternal >= SAMPLE_SPLITTING_ID)
    {
        // extract the k value
        int k = BestCompression.CompressionIdentifierInternal - SAMPLE_SPLITTING_ID;
        // write the word with the k value as number of bits and the value to write is 0
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeValue(device_object->OutputDataBlock, 0 , BestCompression.data[i] >> k);
            writeValue(device_object->OutputDataBlock, 1, 1);
        }

        // write the word
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeWord(device_object->OutputDataBlock, BestCompression.data[i], k);
        }
    }
    else
    {
        printf("Error: Unknown Compression Identifier\n");
    }

    // Freeing BestCompression method data field, since we made a copy.
    free(BestCompression.data);
    
    // Packed array now contains the compressed block
    PRINT_AEE_COMPRESSED_ARRAY(FinalProcessedBlock);
    
    /* 
    We use an long unsigned long array (32 bit per pos), however, the packed array - max - size equals:
        Max packet size = 6 bits selected option field + 16 bits size field + (n_bits * J_BlockSize) packed array size
    */
    if (CompressionTechniqueIdentifier == 0) {
		//second extension option
		printf("ZeroBlock\n");
	}
    else if (CompressionTechniqueIdentifier == 0xF) {
		//no compression
		printf("no compression\n");
	}
	else {
		//k-split
		printf("k-split %d\n", CompressionTechniqueIdentifier-1);
	}
}


