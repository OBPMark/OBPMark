#ifndef FSPACK_H
#define FSPACK_H

#include "Config.h"
#include "lib_functions.h"

/* FS Debug log, uncomment for debug logging */
// #define FS_PRINT_STATE

#ifdef FS_PRINT_STATE
  #define FS_PRINT(a) printf a
#else
  #define FS_PRINT(a) (void)0
#endif

/* Preprocesses the data for the algorithm */
struct FCompressedData FundamentalSequence(unsigned int* Samples);
/* Writes bits following the corresponding algorithm */
void FundamentalSequenceWriter(struct DataObject* device_object, struct FCompressedData* BestCompression);

#endif