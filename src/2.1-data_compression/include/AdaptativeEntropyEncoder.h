#ifndef AEE_H
#define AEE_H

#include "Config.h"
#include "lib_functions.h"


// Note: the print debug is only safe on non-parallel operations
// #define AEE_DEBUG 
// #define AEE_PRINT_STATE


#ifdef AEE_DEBUG
#define PRINT_AEE_COMPRESSED_ARRAY(x) \
printf("Packet data field: "); \
for(int i = x.size - 1; i >= 0; --i) \
{ \
    printf("%d" ,(x.data[i/32] & (1 << (i%32) )) != 0); \
} \
printf("\n")
#else
#define PRINT_AEE_COMPRESSED_ARRAY(x) 
#endif

#ifdef AEE_PRINT_STATE
  #define AEE_PRINT(a) printf a
#else
  #define AEE_PRINT(a) (void)0
#endif


void AdaptativeEntropyEncoder(struct DataObject* device_object , unsigned int* Samples, struct ZeroBlockProcessed ZeroNum);

#endif