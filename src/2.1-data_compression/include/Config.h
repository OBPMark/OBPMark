#ifndef CONFIG_H
#define CONFIG_H

#include <math.h>

/* Number of iterations */
#define STEPS 1

/* Number of bits of each sample: Resolution. Max: 32 */
#define n_bits 16

/* Number of samples per Block: Block Size. Can be: 8, 16, 32, 64 */
#define J_BlockSize 16

/* Number of blocks */
#define r_samplesInterval 16

/* Even if we use a fixed unsigned long int array, the size of every block equals to the number of bits per number of samples in each block */
#define non_compressed_size (n_bits * J_BlockSize)

/* CUDA DEFINES */
// CUDA BLOCK SIZE
#define BLOCK_SIZE  16
// NUMER OF STREAMS THE APLICATION WILL USE
#define NUMBER_STREAMS 4


/* Is the preprocessor active */ 
#define PREPROCESSOR_ACTIVE // fixmevori: Unstable - WIP, might need some clearance, discuss with CCSDS experts. KQ: Refenrece sample?


/* RANDOM DATA GENERATION */
#define RANDOM_DATA_GENERATION false
// Note: the print debug is only safe on non-parallel operations
// #define CONFIG_DEBUG 
// #define CONFIG_PRINT_STATE


/* Ensure definition */
#define ensure(x) if(!(x)) { printf("Ensure error in %s @ line %d: \"%s\" wasn't ensured.\n", __FILE__, __LINE__, #x); exit(0); }


#ifdef CONFIG_DEBUG
#define PRINT_HEADER(x) \
printf("Header data field: "); \
for(int i = 5; i >= 0; --i) \
{ \
    printf("%d" ,(x & (1 << (i%8) )) != 0); \
} \
printf("\n")
#else
#define PRINT_HEADER(x) 
#endif


#ifdef CONFIG_PRINT_STATE
  #define CONFIG_PRINT(a) printf a
#else
  #define CONFIG_PRINT(a) (void)0
#endif


#define ZERO_BLOCK_ID 0
#define FUNDAMENTAL_SEQUENCE_ID 1
#define SECOND_EXTENSION_ID 2
#define SAMPLE_SPLITTING_ID 3
#define NO_COMPRESSION_ID 32


/* Return data type for compression algorithms */
struct FCompressedData 
{
    unsigned int size;
    unsigned int* data;
    unsigned char CompressionIdentifier; 
    unsigned int CompressionIdentifierInternal;
};

/* 
This struct stores the last position a zeroblock presented a 0 including the number of 0's it presented 
position = -1; <- invalid position
*/
struct ZeroBlockCounter 
{
    unsigned int counter;
    int position;
};

/* Processed struct that determines the number of 0 to set for a given block, -1 means that the block is not a zero mem block */
struct ZeroBlockProcessed
{
    int NumberOfZeros;
};

#endif
