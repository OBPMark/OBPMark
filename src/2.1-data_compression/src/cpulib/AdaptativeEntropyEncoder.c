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
    // Data preprocessing
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

    // If the selected technique is Zero Block or the Second Extension, the compression_technique_identifier_size is +1
    if(BestCompression.CompressionIdentifierInternal == ZERO_BLOCK_ID || BestCompression.CompressionIdentifierInternal == SECOND_EXTENSION_ID)
    {
        compression_technique_identifier_size += 1;
    }
    
    // The sizeof an unsigned short is 16 bit. We simply can use this conversion method to have a good bit stream for this field.
    const unsigned char CompressionTechniqueIdentifier = BestCompression.CompressionIdentifier;
    
    // Write the header
    writeWordChar(device_object->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);
    
    // Write the compressed blocks based on the selected compression algorithm
    if (BestCompression.CompressionIdentifierInternal == ZERO_BLOCK_ID)
    {
        ZeroBlockWriter(device_object, &BestCompression);
    }
    else if(BestCompression.CompressionIdentifierInternal == NO_COMPRESSION_ID)
    {
        NoCompressionWriter(device_object, &BestCompression);
    }
    else if(BestCompression.CompressionIdentifierInternal == FUNDAMENTAL_SEQUENCE_ID)
    {
        FundamentalSequenceWriter(device_object, &BestCompression);
    }
    else if(BestCompression.CompressionIdentifierInternal == SECOND_EXTENSION_ID)
    {
        SecondExtensionWriter(device_object, &BestCompression);
    }
    else if(BestCompression.CompressionIdentifierInternal >= SAMPLE_SPLITTING_ID)
    {
        SampleSplittingWriter(device_object, &BestCompression);
    }
    else
    {
        AEE_PRINT(("Error: Unknown Compression Identifier\n"));
    }

    // Freeing BestCompression method data field, since we made a copy.
    free(BestCompression.data);
}


