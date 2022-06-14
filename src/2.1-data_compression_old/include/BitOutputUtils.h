#ifndef BITOUTPUTUTILS_H
#define BITOUTPUTUTILS_H


struct OutputBitStream
{
    unsigned char* OutputBitStream;
    unsigned int num_bits;
    unsigned int num_total_bytes;
}; 


// Writes to the OutputBitStream the variable word with the number of bits
void writeWord(struct OutputBitStream *status, unsigned int word, int number_bits);
void writeWordChar(struct OutputBitStream *status, unsigned char word, int number_bits);
// Writes to the OutputBitStream the variable value (that is a bit) with the number of bits
void writeValue(struct OutputBitStream *status, unsigned char value, int number_bits);


#endif // BITOUTPUTUTILS_H