#ifndef NC_H
#define NC_H

#include "Config.h"
#include "BitOutputUtils.h"


//#define NC_DEBUG 
//#define NC_PRINT_STATE


#ifdef NC_DEBUG
#define PRINT_NC_COMPRESSED_ARRAY(x) \
printf("Compressed bit array: "); \
for(int i = (J_BlockSize*32) - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#else
#define PRINT_NC_COMPRESSED_ARRAY(x) 
#endif

#ifdef NC_PRINT_STATE
  #define NC_PRINT(a) printf a
#else
  #define NC_PRINT(a) (void)0
#endif


struct FCompressedData NoCompression(unsigned int* Samples);

#endif