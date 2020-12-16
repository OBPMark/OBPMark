#ifndef SEPACK_H
#define SEPACK_H

#include "Config.h"

/* SE Debug log, uncomment for debug logging */
// #define SE_DEBUG 
// #define SE_PRINT_STATE

#ifdef SE_DEBUG
#define PRINT_SE_COMPRESSED_ARRAY(x) \
printf("Compressed bit array: "); \
for(int i = (J_BlockSize*32) - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#define PRINT_HALVED_ARRAY_ELEMENT(i,el) printf("HalvedSample %d: %ld\n", i, el)
#else
#define PRINT_SE_COMPRESSED_ARRAY(x) 
#define PRINT_HALVED_ARRAY_ELEMENT(i,el)
#endif

#ifdef SE_PRINT_STATE
  #define SE_PRINT(a) printf a
#else
  #define SE_PRINT(a) (void)0
#endif


// This algorithm splits in half the blocksize
#define HalfBlockSize J_BlockSize/2

/* Returns the processed size */
struct FCompressedData SecondExtension(unsigned long int* Samples);

#endif