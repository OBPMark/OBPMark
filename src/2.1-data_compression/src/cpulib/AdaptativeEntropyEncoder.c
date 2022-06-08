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


struct FCompressedData AdaptativeEntropyEncoder(struct DataObject* device_object , unsigned int* Samples, struct ZeroBlockProcessed ZeroNum)
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
    
    

    struct FCompressedData FinalProcessedBlock;
    
    unsigned int compression_technique_identifier_size = 1;
    // define the size of the compression technique identifier base of n_bits size
    if (n_bits < 3){
        compression_technique_identifier_size = 1;
    }
    else if (n_bits < 5)
    {
        compression_technique_identifier_size = 2;
    }
    else if (n_bits > 4 && n_bits <= 8)
    {
        compression_technique_identifier_size = 3;
    }
    else if (n_bits > 8 && n_bits <= 16)
    {
        compression_technique_identifier_size = 4;
    }
    else if (n_bits > 16 && n_bits <= 32)
    {
        compression_technique_identifier_size = 5;
    }
    else {
        compression_technique_identifier_size = 5;
    }

    // The final size equals 6 bits for the selected option + 16 bits for the size field-
    FinalProcessedBlock.size = compression_technique_identifier_size + BestCompression.size;
    FinalProcessedBlock.CompressionIdentifier = BestCompression.CompressionIdentifier;
    
    /* For the compression technique identifier we are going to use 6 bits to encode the selected option
        000000 : ZeroBlock
        000001 : Second Extension
        000010 + k: Sample Splitting. Including k = 0
        100000 : No compression
    */
    
    
    // The sizeof an unsigned short is 16 bit. We simply can use this conversion method to have a good bit stream for this field.
    const unsigned char CompressionTechniqueIdentifier = BestCompression.CompressionIdentifier;
    // This array will return the standarized output for the AEE
    FinalProcessedBlock.data = calloc(J_BlockSize, sizeof(unsigned int));  
    //FinalProcessedBlock.data[0] = CompressionTechniqueIdentifier << (compression_technique_identifier_size);
    printf("Compression Technique Identifier: %d\n", CompressionTechniqueIdentifier);
    // if the selected technique is the zero block or the second extension, the compression_technique_identifier_size is +1
    if(CompressionTechniqueIdentifier == ZERO_BLOCK_ID || CompressionTechniqueIdentifier == SECOND_EXTENSION_ID)
    {
        compression_technique_identifier_size += 1;
    }
    writeWordChar(device_object->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);
    
    // write each bit of the compressed data
    
    //printf("%d", BestCompression.data[i]);

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
            writeWord(device_object->OutputDataBlock,  1, BestCompression.data[i]+1);
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
        //printf("k: %d\n", k);
        //printf("%u %u\n", BestCompression.data[i], BestCompression.data[i] >> k);
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeValue(device_object->OutputDataBlock, 0 , BestCompression.data[i] >> k);
            // write 1 with 1 bit 
            writeValue(device_object->OutputDataBlock, 1, 1);

        }

        // write the word
        //
        for(int i = 0; i < J_BlockSize; ++i)
        {
            writeWord(device_object->OutputDataBlock, BestCompression.data[i], k);
        }


    }
    else
    {
        printf("Error: Unknown Compression Identifier\n");
    }

    // now we have to copy the data from the compressed data to the final output
    // we need to take into account the size of the compression technique identifier
    // the BestCompression.size is the size of the compressed data in bits
    // the compression_technique_identifier_size is the size of the compression technique identifier in bits
    
    // Bit shifting compressed array compression_technique_identifier_size bits to input the header (size and identification method).
    /*for(int i = 0; i < 32 * J_BlockSize; ++i)
    {
        // runs for the size of the compressed data in bits, now flip the bits and shift them to the left for each element base on the size of the compress selection

        if((BestCompression.data[i/32] & (1 << (i%32) )) != 0)
        {
            FinalProcessedBlock.data[(i+compression_technique_identifier_size)/32] |= 1 << ((i+compression_technique_identifier_size)%32);
        }
    }*/




    // Bit shifting compressed array compression_technique_identifier_size bits to input the header (size and identification method). 
            
    // take BestCompression.data and copy it to the FinalProcessedBlock.data with switching the order of the bytes
    
    //FinalProcessedBlock.data[0] = BestCompression.data[0];

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
    return FinalProcessedBlock;
}


