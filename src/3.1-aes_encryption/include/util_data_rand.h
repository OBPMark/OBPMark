/**
 * \file util_data_rand.h 
 * \brief Benchmark #3.1 AES Encryption
 * \author Marc Solé Bonet (BSC)
 */
#ifndef UTIL_DATA_RAND_H_
#define UTIL_DATA_RAND_H_


#include "obpmark.h"
#include "benchmark.h"

#define BENCHMARK_RAND_SEED	21121993

void benchmark_gen_rand_data(uint8_t *input_data, uint8_t *cypher_key, unsigned int data_length, unsigned int key_length);

#endif // UTIL_DATA_RAND_H_
