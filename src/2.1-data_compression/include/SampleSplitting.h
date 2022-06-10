#ifndef SSPACK_H
#define SSPACK_H

#include "Config.h"
#include "lib_functions.h"

/* SS Debug log, uncomment for debug logging */
// #define SS_PRINT_STATE

#ifdef SS_PRINT_STATE
  #define SS_PRINT(a) printf a
#else
  #define SS_PRINT(a) (void)0
#endif

/* Preprocesses the data for the algorithm */
struct FCompressedData SampleSplitting(unsigned int* Samples, unsigned int k);
/* Writes bits following the corresponding algorithm */
void SampleSplittingWriter(struct DataObject* device_object, struct FCompressedData* BestCompression);

#endif