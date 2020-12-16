#ifndef FSPACK_H
#define FSPACK_H

#include "Config.h"

/* FS Debug log, uncomment for debug logging */
// #define FS_DEBUG 
// #define FS_PRINT_STATE


#ifdef FS_DEBUG
#define PRINT_FS_COMPRESSED_ARRAY(x) \
printf("Compressed bit array: "); \
for(int i = (J_BlockSize*32) - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#else
#define PRINT_FS_COMPRESSED_ARRAY(x) 
#endif

#ifdef FS_PRINT_STATE
  #define FS_PRINT(a) printf a
#else
  #define FS_PRINT(a) (void)0
#endif

/* Returns the processed size */
struct FCompressedData FundamentalSequence(unsigned long int* Samples);

#endif