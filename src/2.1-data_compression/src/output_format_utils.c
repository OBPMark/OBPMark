#include "output_format_utils.h"


void writeWord(struct OutputBitStream *status, unsigned int word, int number_bits)
{
    for (int i = number_bits - 1; i >=0; --i)
    {
        status->OutputBitStream[status->num_total_bytes] |= ((word >>i) & 0x1) << (7 - status->num_bits);
        status->num_bits++;
        // Check if the byte is complete
        if (status->num_bits >= 8)
        {
            status->num_bits = 0;
            status->num_total_bytes++;
        }
    }
}


void writeWordChar(struct OutputBitStream *status, unsigned char word, int number_bits)
{
    for (int i = number_bits - 1; i >=0; --i)
    {
        status->OutputBitStream[status->num_total_bytes] |= ((word >>i) & 0x1) << (7 - status->num_bits);
        status->num_bits++;
        // Check if the byte is complete
        if (status->num_bits >= 8)
        {
            status->num_bits = 0;
            status->num_total_bytes++;
        }
    }
}


void writeValue(struct OutputBitStream *status, unsigned char value, int number_bits)
{
    // Check if value is 0 or 1 if not return
    if (value != 0 && value != 1)
    {
        return;
    }
    
    value = 0x1 & value;
    for (unsigned int i = 0; i < number_bits; ++i)
    {
        status->OutputBitStream[status->num_total_bytes] |= value << (7 - status->num_bits);
        status->num_bits++;
        // Check if the byte is complete
        if (status->num_bits >= 8)
        {
            status->num_bits = 0;
            status->num_total_bytes++;
        }
    }
}


void ZeroBlockWriter (struct OutputBitStream *status, unsigned int size)
{
    writeValue(status,  0, size - 1);
    writeValue(status, 1,1);
}

void NoCompressionWriter(struct OutputBitStream *status,unsigned int  j_blocksize, unsigned int n_bits, unsigned int* data)
{
    for(int i = 0; i < j_blocksize; ++i)
    {
        writeWord(status,  data[i], n_bits);
    }
}

void FundamentalSequenceWriter(struct OutputBitStream *status ,unsigned int  j_blocksize,  unsigned int* data)
{
    for(int i = 0; i < j_blocksize; ++i)
    {
        writeValue(status, 0 , data[i]);
        writeValue(status, 1, 1);
    }
}

void SecondExtensionWriter(struct OutputBitStream *status, unsigned int  HalfBlockSize, unsigned int* data)
{
    for(int i = 0; i < HalfBlockSize; ++i)
    {
        writeWord(status,  data[i], sizeof(unsigned int) * 8);
    }
}

void SampleSplittingWriter(struct OutputBitStream *status, unsigned int  j_blocksize, unsigned int k, unsigned int* data)
{
    // MSB shifted k right dictates the 0 to write + a one (following fundamental sequence)
    for(int i = 0; i < j_blocksize; ++i)
    {
        writeValue(status, 0 , data[i] >> k);
        writeValue(status, 1, 1);
    }
    // Append the LSB part
    for(int i = 0; i < j_blocksize; ++i)
    {
        writeWord(status, data[i], k);
    }
}
