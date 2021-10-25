/**
 * \file aes.h
 * \author David Steenari (ESA)
 * \brief AES encryption implementation for OBPMark.
 */
#ifndef OBPMARK_AES_H_
#define OBPMARK_AES_H_

#include "obpmark.h"

/**
 * \brief Allowed AES key lengths.
 */
typedef enum {
	AES_KEY128 = 128,
	AES_KEY192 = 192,
	AES_KEY256 = 256
} AES_keysize_t;

/**
 * \brief Encrypts a buffer of data with AES. 
 * \param key		Encryption key. 
 * \param key_size 	Size of key in bits, shall be 128, 192 or 256 bits. 
 * \param data		Input plaintext data buffer of length "data_length".
 * \param data_length	Length of data buffer to encrypt. 
 * \param out		OUtput buffer to store encrypted ciphertext. 
 */
void AES_encrypt(uint32_t *key, unsigned int key_size, uint8_t *data, size_t data_length, uint8_t *out);

#endif // OBPMARK_AES_H_
