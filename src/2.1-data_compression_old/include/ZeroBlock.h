#ifndef ZBPACK_H
#define ZBPACK_H

#include "Config.h"
#include "lib_functions.h"

/* SE Debug log, uncomment for debug logging */
// #define ZB_PRINT_STATE

#ifdef ZB_PRINT_STATE
  #define ZB_PRINT(a) printf a
#else
  #define ZB_PRINT(a) (void)0
#endif

/* Preprocesses the data for the algorithm */
struct FCompressedData ZeroBlock(unsigned int* Samples, unsigned int NumberOfZeros);
/* Writes bits following the corresponding algorithm */
void ZeroBlockWriter(struct DataObject* device_object, struct FCompressedData* BestCompression);

#endif