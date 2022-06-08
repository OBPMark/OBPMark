#ifndef BITOUTPUTUTILS_H
#define BITOUTPUTUTILS_H



#define FALSE 0
#define TRUE 1

struct OutputBitStream
{
    unsigned char* OutputBitStream;
    unsigned int num_bits;
    unsigned int num_total_bytes;
}; 


// write to the OutputBitStream the variable word with the number of bits
void writeWord(struct OutputBitStream *status, unsigned int word, int number_bits);
void writeWordChar(struct OutputBitStream *status, unsigned char word, int number_bits);
// write to the OutputBitStream the variable value (that is a bit) with the number of bits
unsigned int writeValue(struct OutputBitStream *status, unsigned char value, int number_bits);




#endif // BITOUTPUTUTILS_H