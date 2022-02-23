/**
 * \file util_data_rand.c 
 * \brief Benchmark #3.1 AES Encryption
 * \author Marc Solé Bonet (BSC)
 */

#include "util_data_rand.h"

void benchmark_gen_rand_data(uint8_t *input_data, uint8_t *cypher_key, unsigned int data_length, unsigned int key_length)
{

	/* Initialize srad seed */
	srand(BENCHMARK_RAND_SEED);
	// DEFAULT 8 bits
    int	randnumber = 255;
	
	/* Input text */
	for(int i = 0; i < data_length; i++) 
	    input_data[i] = (uint8_t) rand() % randnumber;
	for(int i = 0; i < key_length; i++)
	    cypher_key[i] = (uint8_t) rand() % randnumber;

}
