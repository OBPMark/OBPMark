/** 
 * \brief BPMark "Data compression algorithm." processing task and image kernels.
 * \file processing.c
 * \author Ivan Rodriguez-Ferrandez (BSC)
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "processing.h"
#include "obpmark.h"
#include "obpmark_time.h"

/* preprocessor section */
int DelayedStack = 0;

int UnitDelayPredictor(int DataSample) 
{
    const int CachedDelayedStack = DelayedStack;
    DelayedStack = DataSample;
    return CachedDelayedStack;
}

int PredictorErrorMapper(int PredictedValue, int PredictionError, unsigned int n_bits)
{
    const int x_min = 0;
    const int x_max = pow( 2,n_bits) - 1;

    const int theta = min(PredictedValue - x_min, x_max-PredictedValue);
    int PreprocessedSample = theta + abs(PredictionError);

    if(0 <= PredictionError && PredictionError <= theta)
    {
        PreprocessedSample = 2 * PredictionError;
    }
    else if(-theta <= PredictionError && PredictionError < 0)
    {
        PreprocessedSample = (2 * abs(PredictionError)) - 1;
    }

    return PreprocessedSample;
}

int Preprocessor(int x, unsigned int n_bits){
    const int PredictedValue = UnitDelayPredictor(x);
    const int PredictionError = x - PredictedValue;
    const int PreprocessedSample = PredictorErrorMapper(PredictedValue, PredictionError,n_bits);
    return PreprocessedSample;
}

/* end preprocessor section */


/* adaptive entropy encoder section */

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
/* No compression part */
struct FCompressedData NoCompression(compression_data_t *compression_data, unsigned int* Samples)
{
    // Output size sanitization
    struct FCompressedData CompressedData;
    CompressedData.CompressionIdentifierInternal = NO_COMPRESSION_ID;
    CompressedData.size = compression_data->n_bits * compression_data->j_blocksize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * compression_data->j_blocksize);
    memcpy(CompressedData.data, Samples, sizeof(unsigned int)* compression_data->j_blocksize);

    if (compression_data->n_bits < 3)
    {
        CompressedData.CompressionIdentifier = 0x1; //1
    }
    else if (compression_data->n_bits < 5)
    {
        CompressedData.CompressionIdentifier = 0x3; //11
    }
    else if (compression_data->n_bits <= 8)
    {
        CompressedData.CompressionIdentifier = 0x7; //111 
    }
    else if (compression_data->n_bits <= 16)
    {
        CompressedData.CompressionIdentifier = 0xF; //1111 
    }
    else /*if (compression_data->n_bits <= 32)*/
    {
        CompressedData.CompressionIdentifier = 0x1F; //11111 
    }

    return CompressedData;
}


/* Second extension */


unsigned int GetSizeSecondExtension(unsigned int HalfBlockSize, unsigned int* HalvedSamples)
{
    unsigned int size = 0;
    for(int i = 0; i < HalfBlockSize; ++i)
    {
        size += HalvedSamples[i] + 1;
    }
    return size;
}


struct FCompressedData SecondExtension(compression_data_t *compression_data, unsigned int* Samples)
{
    
    const unsigned int HalfBlockSize = compression_data->j_blocksize / 2;
    unsigned int* HalvedSamples = (unsigned int*) malloc (sizeof( unsigned int ) * HalfBlockSize);
    
    // Halving the data using the SE Option algorithm. See: https://public.ccsds.org/Pubs/121x0b2ec1.pdf
    for(unsigned int i = 0; i < HalfBlockSize; ++i)
    {
        HalvedSamples[i] = (((Samples[2*i] + Samples[2*i + 1]) * (Samples[2*i] + Samples[2*i + 1] + 1)) / 2) + Samples[2*i + 1];
    }

    struct FCompressedData CompressedData;
    CompressedData.size = compression_data->j_blocksize * 32;
    CompressedData.data = NULL;
    CompressedData.CompressionIdentifier = SECOND_EXTENSION_ID - 1;

    // Checks if the compressed size is minor than the uncompressed size 
    const unsigned int CompressedSize = GetSizeSecondExtension(HalfBlockSize,HalvedSamples);
    if(CompressedSize > compression_data->j_blocksize * 32)
    {
        return CompressedData;
    }
    
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * compression_data->j_blocksize);
    CompressedData.CompressionIdentifierInternal = SECOND_EXTENSION_ID;
    memcpy(CompressedData.data, HalvedSamples, sizeof( unsigned int ) * HalfBlockSize);
    // free HalvedSamples
    free(HalvedSamples);
    
    return CompressedData;
}
/* Fundamental Sequence */
unsigned int GetSizeFundamentalSequence(unsigned int j_blocksize, unsigned int* Samples)
{
    unsigned int size = 0;
    for(int i = 0; i < j_blocksize; ++i)
    {
        size += Samples[i] + 1;
    }
    return size;
}

struct FCompressedData FundamentalSequence(compression_data_t *compression_data, unsigned int* Samples)
{   
    struct FCompressedData CompressedData;
    CompressedData.size = compression_data->j_blocksize * 32;
    CompressedData.data = NULL;
    CompressedData.CompressionIdentifier = FUNDAMENTAL_SEQUENCE_ID;

    // Checks if the compressed size is minor than the uncompressed size 
    const unsigned int CompressedSize = GetSizeFundamentalSequence(compression_data->j_blocksize, Samples);
    if(CompressedSize > compression_data->j_blocksize * 32)
    {
        return CompressedData;
    }

    CompressedData.size = CompressedSize;
    CompressedData.CompressionIdentifierInternal = FUNDAMENTAL_SEQUENCE_ID;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * compression_data->j_blocksize);
    memcpy(CompressedData.data, Samples, sizeof( unsigned int ) * compression_data->j_blocksize);
    
    return CompressedData;
}

/* Sample splitting */


unsigned int GetSizeSampleSplitting(unsigned int j_blocksize, unsigned int* Samples, unsigned int k)
{
    unsigned int size = 0;
    for(unsigned int i = 0; i < j_blocksize; ++i)
    {
        size += (k + (Samples[i] >> k) + 1);
    }
    return size;
}


struct FCompressedData SampleSplitting(compression_data_t *compression_data, unsigned int* Samples, unsigned int k)
{
    struct FCompressedData CompressedData;
    CompressedData.size = compression_data->j_blocksize * 32;
    CompressedData.data = NULL;
    
    // k sanitization
    if(k >= compression_data->n_bits - 2)
    {
        CompressedData.CompressionIdentifier = SAMPLE_SPLITTING_ID + k;
        return CompressedData;
    }
    else if(k == 0)
    {
        CompressedData.CompressionIdentifier = 1;
        return FundamentalSequence(compression_data, Samples);
    }

    // Output size sanitization
    const unsigned int CompressedSize = GetSizeSampleSplitting(compression_data->j_blocksize, Samples, k);
    if(CompressedSize > compression_data->j_blocksize * 32)
    {
        CompressedData.CompressionIdentifier = SAMPLE_SPLITTING_ID + k;
        return CompressedData;
    }

    CompressedData.CompressionIdentifier = k + 1;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * compression_data->j_blocksize);
    CompressedData.CompressionIdentifierInternal = SAMPLE_SPLITTING_ID + k;
    memcpy(CompressedData.data, Samples, sizeof( unsigned int ) * compression_data->j_blocksize);


    return CompressedData;

}

/* Zero block */

struct FCompressedData ZeroBlock(compression_data_t *compression_data, unsigned int* Samples, unsigned int NumberOfZeros)
{
    // Output size sanitization
    const unsigned int CompressedSize = NumberOfZeros + 1;

    //unsigned int PackedArray[compression_data->j_blocksize] = { 0 };
    // make a calloc for PackedArray
    unsigned int* PackedArray = (unsigned int*) calloc (sizeof( unsigned int ), compression_data->j_blocksize);
    PackedArray[0] = 1;

    struct FCompressedData CompressedData;
    CompressedData.size = CompressedSize;
    CompressedData.data = (unsigned int*) malloc (sizeof( unsigned int ) * compression_data->j_blocksize);
    memcpy(CompressedData.data, PackedArray, sizeof(PackedArray));
    CompressedData.CompressionIdentifier = ZERO_BLOCK_ID;
    CompressedData.CompressionIdentifierInternal = ZERO_BLOCK_ID;

    free(PackedArray);

    return CompressedData;
}


/* final adaptive entropy encoder section */
void AdaptativeEntropyEncoder(
    compression_data_t *compression_data,
    unsigned int * Samples,
    struct ZeroBlockProcessed *ZeroNum)
{
    // Data preprocessing
    struct FCompressedData BestCompression;
    if(ZeroNum->NumberOfZeros == -1)
    {
        const struct FCompressedData size_no_compression = NoCompression(compression_data, Samples);
        const struct FCompressedData size_se  = SecondExtension(compression_data, Samples);
        
        BestCompression = MIN(size_no_compression, size_se);
        
        // Sample splitting k = i
        for(int i = 0; i < compression_data->n_bits; ++i)
        {
            BestCompression = MIN(SampleSplitting(compression_data, Samples, i), BestCompression);
        }  
    }
    else
    {
       BestCompression = ZeroBlock(compression_data, Samples, ZeroNum->NumberOfZeros);
    }

    // now prepare to write the compressed data
    unsigned int compression_technique_identifier_size = 1;
    // define the size of the compression technique identifier base of n_bits size
    if (compression_data->n_bits < 3){
        compression_technique_identifier_size = 1;
    }
    else if (compression_data->n_bits < 5)
    {
        compression_technique_identifier_size = 2;
    }
    else if (compression_data->n_bits <= 8)
    {
        compression_technique_identifier_size = 3;
    }
    else if (compression_data->n_bits <= 16)
    {
        compression_technique_identifier_size = 4;
    }
    else /*if (compression_data->n_bits <= 32)*/
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
    writeWordChar(compression_data->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);

    // Write the compressed blocks based on the selected compression algorithm
    if (BestCompression.CompressionIdentifierInternal == ZERO_BLOCK_ID)
    {
        ZeroBlockWriter(compression_data->OutputDataBlock, BestCompression.size);
    }
    else if(BestCompression.CompressionIdentifierInternal == NO_COMPRESSION_ID)
    {
        NoCompressionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, compression_data->n_bits, BestCompression.data);
    }
    else if(BestCompression.CompressionIdentifierInternal == FUNDAMENTAL_SEQUENCE_ID)
    {
        FundamentalSequenceWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, BestCompression.data);
    }
    else if(BestCompression.CompressionIdentifierInternal == SECOND_EXTENSION_ID)
    {
        SecondExtensionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize/2,BestCompression.data);
    }
    else if(BestCompression.CompressionIdentifierInternal >= SAMPLE_SPLITTING_ID)
    {
        SampleSplittingWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, BestCompression.CompressionIdentifierInternal - SAMPLE_SPLITTING_ID, BestCompression.data);
    }
    else
    {
        printf("Error: Unknown compression technique identifier\n");
    }

    // Freeing BestCompression method data field, since we made a copy.
    free(BestCompression.data);
}


/* end adaptive entropy encoder section */

void preprocess_data(
	compression_data_t *compression_data,
	bool *AllZerosInBlock,
	unsigned int *ZeroCounterPos,
	struct ZeroBlockCounter * ZeroCounter,
    unsigned int step
	)
{
    // Pre-processing the input data values, precalculating ZeroBlock offsets
        for(unsigned int block = 0; block < compression_data->r_samplesInterval; ++block)
        {
            
            // Preprocessing the samples 
            for(unsigned int i = 0; i < compression_data->j_blocksize; ++i)
            {
                if(compression_data->preprocessor_active)
                {
                    compression_data->OutputPreprocessedValue[i + (block * compression_data->j_blocksize)] = Preprocessor(compression_data->InputDataBlock[(i + (block * compression_data->j_blocksize)) + (step * compression_data->r_samplesInterval * compression_data->j_blocksize)], compression_data->n_bits);
                }
                else
                {
                    compression_data->OutputPreprocessedValue[i + (block * compression_data->j_blocksize)] = compression_data->InputDataBlock[(i + (block * compression_data->j_blocksize)) + (step * compression_data->r_samplesInterval * compression_data->j_blocksize)];
                }
                if (compression_data->OutputPreprocessedValue[i + (block * compression_data->j_blocksize)] != 0) *AllZerosInBlock = false;
            }
            // Zero Block post processing
            if( *AllZerosInBlock == true )
            {
                // Increasing the zero count in the record, we found another all-zero block
                ZeroCounter[*ZeroCounterPos].counter++;
                ZeroCounter[*ZeroCounterPos].position = block;

            }
            else if(ZeroCounter[*ZeroCounterPos].position != -1)
            {
                // Increasing ZeroCounterPos only if we found a non zero numbers
                *ZeroCounterPos++;
                *AllZerosInBlock = true;
            }
        }
}

void process_zeroblock(
	compression_data_t *compression_data,
	bool *AllZerosInBlock,
	unsigned int *ZeroCounterPos,
	struct ZeroBlockCounter *ZeroCounter,
	struct ZeroBlockProcessed *ZBProcessed
	)
{
    // Processing the ZeroBlock Arrays: Adding the number of 0's to be written per block
    for(unsigned int i = 0; i < *ZeroCounterPos; ++i)
    {        
        for(unsigned int pos = 0; pos < ZeroCounter[i].counter; ++pos)
        {
            // Calculating ROS (Remainder of a segment)
            const short int bSkip5th = !((ZeroCounter[i].counter < 9 && (ZeroCounter[i].position - pos) >= 4) || (ZeroCounter[i].counter >= 9 && (ZeroCounter[i].position - pos) >= 5));
            const unsigned int z_number = ZeroCounter[i].counter - pos - bSkip5th;
            ZBProcessed[ZeroCounter[i].position - pos].NumberOfZeros = z_number;
        }
    }
}




void process_blocks(
	compression_data_t *compression_data,
	struct ZeroBlockProcessed *ZBProcessed
	)
{
     for(unsigned int block = 0; block < compression_data->r_samplesInterval; ++block)
    {    
        AdaptativeEntropyEncoder(compression_data, compression_data->OutputPreprocessedValue + (compression_data->j_blocksize*block), &ZBProcessed[block]);
    }
}