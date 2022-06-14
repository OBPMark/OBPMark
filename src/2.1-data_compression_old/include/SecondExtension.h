#ifndef SEPACK_H
#define SEPACK_H

#include "Config.h"
#include "lib_functions.h"

/* SE Debug log, uncomment for debug logging */
// #define SE_PRINT_STATE

#ifdef SE_PRINT_STATE
  #define SE_PRINT(a) printf a
#else
  #define SE_PRINT(a) (void)0
#endif

// This algorithm splits in half the blocksize
#define HalfBlockSize J_BlockSize/2

/* Preprocesses the data for the algorithm */
struct FCompressedData SecondExtension(unsigned int* Samples);
/* Writes bits following the corresponding algorithm */
void SecondExtensionWriter(struct DataObject* device_object, struct FCompressedData* BestCompression);

#endif