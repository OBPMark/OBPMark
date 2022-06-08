#include "BitOutputUtils.h"





void writeWord(struct OutputBitStream *status, unsigned int word, int number_bits)
{
    for (int i = number_bits - 1; i >=0; --i)
    {
        status->OutputBitStream[status->num_total_bytes] |= ((word >>i) & 0x1) << (7 - status->num_bits);
        status->num_bits++;
        // check if the byte is full
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
        // check if the byte is full
        if (status->num_bits >= 8)
        {
            status->num_bits = 0;
            status->num_total_bytes++;
        }
    }
}

// write to the OutputBitStream the variable value (that is a bit) with the number of bits
unsigned int writeValue(struct OutputBitStream *status, unsigned char value, int number_bits)
{
    // check if value is binary 0 or binary 1 if not return false
    if (value != 0 && value != 1)
    {
        return FALSE;
    }
    value = 0x1 & value;
    for (unsigned int i = 0; i < number_bits; ++i)
    {
        status->OutputBitStream[status->num_total_bytes] |= value << (7 - status->num_bits);
        status->num_bits++;
        // check if the byte is full
        if (status->num_bits >= 8)
        {
            status->num_bits = 0;
            status->num_total_bytes++;
        }

    }
    return TRUE;
}