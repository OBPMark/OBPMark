#ifndef OUTPUTFORMATUTILS_H
#define OUTPUTFORMATUTILS_H


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

// write the different types of data to the OutputBitStream
void ZeroBlockWriter (struct OutputBitStream *status, unsigned int size);
void NoCompressionWriter(struct OutputBitStream *status,unsigned int  j_blocksize, unsigned int n_bits, unsigned int* data);
void FundamentalSequenceWriter(struct OutputBitStream *status ,unsigned int  j_blocksize,  unsigned int* data);
void SecondExtensionWriter(struct OutputBitStream *status, unsigned int  HalfBlockSize, unsigned int* data);
void SampleSplittingWriter(struct OutputBitStream *status, unsigned int  j_blocksize, unsigned int k, unsigned int* data);



#endif // OUTPUTFORMATUTILS_H