#ifndef NC_H
#define NC_H

#include "Config.h"
#include "lib_functions.h"

/* NC Debug log, uncomment for debug logging */
// #define NC_PRINT_STATE

#ifdef NC_PRINT_STATE
  #define NC_PRINT(a) printf a
#else
  #define NC_PRINT(a) (void)0
#endif

/* Preprocesses the data for the algorithm */
struct FCompressedData NoCompression(unsigned int* Samples);
/* Writes bits following the corresponding algorithm */
void NoCompressionWriter(struct DataObject* device_object, struct FCompressedData* BestCompression);

#endif