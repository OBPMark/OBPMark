/**
 * \file processing.h
 * \author David Steenari (ESA)
 * \brief AES encryption implementation for OBPMark.
 */
#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "device.h"
#include <omp.h>

#define STATE_PARAM	uint8_t state[4][4]
#define INSTATE_PARAM uint8_t in_state[4][4]
#define STATES_PARAM INSTATE_PARAM, STATE_PARAM
#define ROUNDKEY_PARAM	uint8_t *roundkey
#define SBOX_PARAM uint8_t *sbox
#define NB_PARAM unsigned int Nb

#define KEY_PARAM AES_key_t *key
#define EXPKEY_PARAM uint32_t *expanded_key
#define RCON_PARAM uint8_t *rcon

#define DATA_PARAM AES_values_t *AES_data
#define TIME_PARAM AES_time_t *t

__global__ void AES_KeyExpansion(DATA_PARAM);
__global__ void AES_encrypt(DATA_PARAM);

#endif // OBPMARK_AES_H_
