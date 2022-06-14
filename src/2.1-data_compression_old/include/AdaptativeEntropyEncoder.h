#ifndef AEE_H
#define AEE_H

#include "Config.h"
#include "lib_functions.h"

// Note: the print debug is only safe on non-parallel operations
// #define AEE_PRINT_STATE

#ifdef AEE_PRINT_STATE
  #define AEE_PRINT(a) printf a
#else
  #define AEE_PRINT(a) (void)0
#endif

/* AEE Algorithm: Selects the best codification algorithm and assembles the data */
void AdaptativeEntropyEncoder(struct DataObject* device_object , unsigned int* Samples, struct ZeroBlockProcessed ZeroNum);

#endif