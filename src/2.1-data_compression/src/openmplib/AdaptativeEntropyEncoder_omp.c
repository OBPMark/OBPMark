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


struct FCompressedData AdaptativeEntropyEncoder(unsigned long int* Samples, struct ZeroBlockProcessed ZeroNum)
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
            BestCompression = MIN(BestCompression, SampleSplitting(Samples, i));
        }  

    }
    else
    {
       BestCompression = ZeroBlock(Samples, ZeroNum.NumberOfZeros);
    }
    
    AEE_PRINT(("Selected Compression Method: %d with %d size.\n", BestCompression.CompressionIdentifier, BestCompression.size));

    /* Once we select the Compression method to apply, we have to encode the output result 
     In this specific case, we decided to encode the output of the AEE the following way:
     [ DATA: Variable length bit ] [ SIZE: 16 bit field ] [ Compression Technique Identification: 6 bit field ]
    */

    struct FCompressedData FinalProcessedBlock;
    
    // The final size equals 6 bits for the selected option + 16 bits for the size field-
    FinalProcessedBlock.size = 22 + BestCompression.size;
    FinalProcessedBlock.CompressionIdentifier = BestCompression.CompressionIdentifier;
    
    /* For the compression technique identifier we are going to use 6 bits to encode the selected option
        000000 : ZeroBlock
        000001 : Second Extension
        000010 + k: Sample Splitting. Including k = 0
        100000 : No compression
    */
    const unsigned char CompressionTechniqueIdentifier = BestCompression.CompressionIdentifier;
    
    // The sizeof an unsigned short is 16 bit. We simply can use this conversion method to have a good bit stream for this field.
    const unsigned short size = BestCompression.size; 
    
    // This array will return the standarized output for the AEE
    FinalProcessedBlock.data = calloc(J_BlockSize, sizeof(unsigned long int));  
    FinalProcessedBlock.data[0] = CompressionTechniqueIdentifier + (size << 6);

    // Bit shifting compressed array 22 bits to input the header (size and identification method).
    for(int i = 0; i < 32 * J_BlockSize; ++i)
    {
        if((BestCompression.data[i/32] & (1 << (i%32) )) != 0)
        {
            FinalProcessedBlock.data[(i+22)/32] |= 1 << ((i+22)%32);
        }
    }

    // Freeing BestCompression method data field, since we made a copy.
    free(BestCompression.data);
    
    // Packed array now contains the compressed block
    PRINT_AEE_COMPRESSED_ARRAY(FinalProcessedBlock);
    
    /* 
    We use an long unsigned long array (32 bit per pos), however, the packed array - max - size equals:
        Max packet size = 6 bits selected option field + 16 bits size field + (n_bits * J_BlockSize) packed array size
    */

    return FinalProcessedBlock;
}


