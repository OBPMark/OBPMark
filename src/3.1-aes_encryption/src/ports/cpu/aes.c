/**
 * \file aes.c
 * \author David Steenari (ESA)
 * \brief AES encryption implementation for OBPMark.
 */
#include "aes.h"

#include <stddef.h>
#include <stdint.h> 

#define STATE_PARAM	uint8_t state[4][4]
#define ROUNDKEY_PARAM	uint8_t roundkey[4][4]


/* Internal function declarations */

void AES_KeyExpansion(AES_keysize_t key_size, uint32_t *key, uint32_t *expanded_key, uint32_t *rcon, uint8_t *rci, unsigned int i); 
void AES_AddRoundKey(STATE_PARAM, ROUNDKEY_PARAM);
void AES_SubBytes(STATE_PARAM, uint8_t *sbox);
void AES_ShiftRows(STATE_PARAM);
void AES_MixColumns(STATE_PARAM);
void AES_encrypt_state(STATE_PARAM, uint8_t *sbox, ROUNDKEY_PARAM, uint32_t *rcon, unsigned int num_rounds);

/* Private functions */

void AES_KeyExpansion(AES_keysize_t key_size, uint32_t *key, uint32_t *expanded_key, uint32_t *rcon, uint8_t *rci, unsigned int i)
{
	// FIXME replace with a lookup table max needed 10: 
	if(i == 1) {
		*rci = 0x01; // FIXME can make a special case for this to remove branch each loop for speedup 
	}
	else if(i > 1) {
		if(*rci < 0x80) {
			*rci = (*rci) << 1; /* rci multiplied by 2 */
		}
		else { /* >= 0x80 */
			*rci = ((*rci) << 1) ^ 0x11;
		}
	}
	else {
		/* Should not happen */
	}
	
	*rcon = *rci;

	// FIXME cleanup above 	
	unsigned int N; /* Key length */ // FIXME put earlier? 
	unsigned int R; /* # of round keys */
	size_t expanded_key_len; 

	switch(key_size) {
		case AES_KEY128: N = 4; R = 11; break;
		case AES_KEY192: N = 6; R = 13; break;
		case AES_KEY256: N = 8; R = 15; break;
	}

	expanded_key_len = 4*R; 

	/* Key expansion */
	// FIXME i should run from 0 ... (4*R - 1)

	/* i < N ==> W_i = K_i */
	for(i=0; i<N; i++) {
		expanded_key[i] = key[i];
	}

	
}

void AES_AddRoundKey(STATE_PARAM, ROUNDKEY_PARAM)
{
	// FIXME add calculation of next roundkey each round 
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
			state[i][j] = state[i][j] ^ roundkey[i][j];
		}
	}
}

void AES_SubBytes(STATE_PARAM, uint8_t *sbox)
{
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
			state[i][j] = sbox[state[i][j]]; 
		}
	}
}

void AES_ShiftRows(STATE_PARAM)
{
	uint8_t temp;
	// FIXME maybe you could do this faster: typecast each row to a u32 and left-shift bitwise

	/* No shift on 1st row */
	/* Shift 2nd row 1 byte */
	temp = state[1][0];
	state[1][0] = state[1][1];
	state[1][1] = state[1][2];
	state[1][2] = state[1][3];
	state[1][3] = temp;

	/* Shift 3rd row 2 bytes */
	temp = state[2][0];
	state[2][0] = state[2][2];
	state[2][2] = temp; 
	temp = state[2][1];
	state[2][1] = state[2][3];
	state[2][3] = temp; 

	/* Shift 4th row 3 bytes (same as 1 byte right shift) */
	temp = state[3][3];
	state[3][3] = state[3][2];
	state[3][2] = state[3][1];
	state[3][1] = state[3][0];
	state[3][0] = temp;
}

void AES_MixColumns(STATE_PARAM)
{
	uint8_t m[4][4] = { 
		{2, 3, 1, 1},
		{1, 2, 3, 1},
		{1, 1, 2, 3},
		{3, 1, 1, 2}
		};

	uint8_t t[4]; 

	/* Columnwise */
	for(int j=0; j<4; j++)
	{
		/* Matrix operation on column vector */
		for(int i=0; i<4; i++)
		{
			// FIXME this multiplicatino will overflow
			// FIXME additions should be XORs
			t[i] = m[i][0] * state[0][j]
			     + m[i][1] * state[1][j]
			     + m[i][2] * state[2][j]
			     + m[i][3] * state[3][j];
		}
		/* Overwrite state */
		state[0][j] = t[0];
		state[1][j] = t[1];
		state[2][j] = t[2];
		state[3][j] = t[3]; 
	} 
}

void AES_encrypt_state(STATE_PARAM, uint8_t *sbox, ROUNDKEY_PARAM, uint32_t *rcon, unsigned int num_rounds) 
{
	/* 1. KeyExpansion */ 
	//AES_KeyExpansion(key_size, key, expanded_key, rcon, rci, i); // FIXME parameters

	/* 2. AddRoundKey */
	// FIXME add calculation of next roundkey each round 
	AES_AddRoundKey(state, roundkey); 

	/* 3. Rounds */
	for(unsigned int roundi=0; roundi<num_rounds; roundi++)
	{
		/* 1. SubBytes 
		 * 2. ShiftRows 
		 * 3. MixColumns 
		 * 4. AddRoundKey */
		AES_SubBytes(state, sbox); 
		AES_ShiftRows(state);
		AES_MixColumns(state);
		AES_AddRoundKey(state, roundkey); 
	}

	/* Final round 
	 * 	- 1. SubBytes
	 * 	- 2. ShiftRows
	 * 	- 3. AddRoundKey */
	AES_SubBytes(state, sbox);
	AES_ShiftRows(state); 
	AES_AddRoundKey(state, roundkey); 
}

/* Public functions */

// FIXME there are more optimized ways of implementing this, e.g. using 32-bit variables for row lookup, and combining steps
// FIXME both x86 and ARM-v8 have specific instructions for implementing an AES round
void AES_encrypt(uint32_t *key, unsigned int key_size, uint8_t *data, size_t data_length, uint8_t *out)
{
	unsigned int x=0, y=0;
	size_t key_len = 0;
	unsigned int num_rounds = 0;
	
	uint8_t state[4][4]; 
	uint8_t sbox[256];
	uint8_t roundkey[4][4]; 

	uint32_t rcon = 0x00000000;
	uint8_t rci = 0x00; 

	/* Set number of words in key buffer and number of rounds depending on key_size */
	if(key_size == AES_KEY128) {
		key_len = 4; 
		num_rounds = 9;
	}
	else if(key_size == AES_KEY192) { 
		key_len = 6;
		num_rounds = 11;
	}
	else if(key_size == AES_KEY256) {
		key_len = 8;
		num_rounds = 13;
	}
	else {
		/* Should not happen */
		return; 
	}

	// FIXME initialize sbox
	// FIXME initalize roundkey 

	while(1) // FIXME exit criteria 
	{
		/* Extract data from buffer to state */
		// FIXME extract data from buf into state
		for(int i=0; i<4; i++) {
			for(int j=0; j<4; j++) {
				state[i][j] = data[x]; // FIXME what about end of file? not multiple of 16?
				x++;
			}
		}

		/* Operations per state */
		AES_encrypt_state(state, sbox, roundkey, &rcon, num_rounds);

		/* Save output */
		for(int i=0; i<4; i++) {
			for(int j=0; j<4; j++) {
				out[y] = state[i][j]; // FIXME what about end of file? not multiple of 16?
				y++;
			}
		}
	}
}

