#ifndef ZBPACK_H
#define ZBPACK_H

#include "Config.h"

/* SE Debug log, uncomment for debug logging */
// #define ZB_DEBUG 
// #define ZB_PRINT_STATE


#ifdef ZB_DEBUG
#define PRINT_ZB_COMPRESSED_ARRAY(x) \
printf("Compressed bit array: "); \
for(int i = (J_BlockSize*32) - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#else
#define PRINT_ZB_COMPRESSED_ARRAY(x) 
#endif

#ifdef ZB_PRINT_STATE
  #define ZB_PRINT(a) printf a
#else
  #define ZB_PRINT(a) (void)0
#endif


/* Returns the processed size */
struct FCompressedData ZeroBlock(unsigned int* Samples, unsigned int NumberOfZeros);

#endif