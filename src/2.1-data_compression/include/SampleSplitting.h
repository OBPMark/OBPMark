#ifndef SSPACK_H
#define SSPACK_H

#include "Config.h"

/* SS Debug log, uncomment for debug logging */
// #define SS_DEBUG 
// #define SS_PRINT_STATE

#ifdef SS_DEBUG
#define PRINT_SS_COMPRESSED_ARRAY(x) \
printf("Compressed bit array: "); \
for(int i = (J_BlockSize*32) - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#else
#define PRINT_SS_COMPRESSED_ARRAY(x) 
#endif

#ifdef SS_PRINT_STATE
  #define SS_PRINT(a) printf a
#else
  #define SS_PRINT(a) (void)0
#endif


/* Returns the processed size */
struct FCompressedData SampleSplitting(unsigned long int* Samples, unsigned int k);

#endif