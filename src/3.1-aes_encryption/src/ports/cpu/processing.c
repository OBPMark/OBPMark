/**
 * \file aes.c
 * \author David Steenari (ESA)
 * \brief AES encryption implementation for OBPMark.
 */
#include "processing.h"

#include <stddef.h>
#include <stdint.h> 
#include <stdio.h>
#include <stdlib.h>


/* Internal function declarations */

void AES_AddRoundKey(STATES_PARAM, ROUNDKEY_PARAM, NB_PARAM, unsigned int round_number);
void AES_SubBytes(STATE_PARAM, SBOX_PARAM);
void AES_ShiftRows(STATE_PARAM);
void AES_MixColumns(STATE_PARAM);
void AES_encrypt_state(STATES_PARAM, NB_PARAM, SBOX_PARAM, ROUNDKEY_PARAM, unsigned int num_rounds);


void printState(STATE_PARAM, int Nb){
    for(int i =0; i<Nb; i++){
        for(int j =0; j<4; j++)
            printf("%#x ", state[j][i]);
        printf("\n");
    }
}
void printKey(ROUNDKEY_PARAM, int round){
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) 
			printf("%#x ", roundkey[(round * 4 * 4) + (j * 4) + i]);
		printf("\n");
	}
}

inline uint32_t AES_RotWord(uint32_t n)
{
    return (n>>8) | (n<<24);
}

uint32_t AES_SubWord(uint32_t word, SBOX_PARAM)
{
    return sbox[word>>24]<<24|sbox[(uint8_t)(word>>16)]<<16|sbox[(uint8_t)(word>>8)]<<8|sbox[(uint8_t)word];
}

void AES_KeyExpansion(KEY_PARAM, EXPKEY_PARAM, SBOX_PARAM, RCON_PARAM)
{
	uint32_t temp;
	int Nk = key->Nk, Nr = key->Nr, Nb = key->Nb;
    int i = 0;
    while(i < Nk){
        expanded_key[i] = *((uint32_t *)&key->value[4*i]);
        i++;
    }
    for(; i < Nb * (Nr+1); i++) {
        temp = expanded_key[i-1];
        if (i%Nk == 0) {
            temp = AES_SubWord(AES_RotWord(temp), sbox) ^ rcon[i/Nk];
        }
        else if (Nk > 6 && i%Nk == 4){
            temp = AES_SubWord(temp,sbox);
        }
        expanded_key[i] = expanded_key[i-Nk] ^ temp;
    }
}

void AES_AddRoundKey(STATES_PARAM, ROUNDKEY_PARAM, NB_PARAM,  unsigned int round_number)
{
	for(int i=0; i<Nb; i++) {
		for(int j=0; j<4; j++) {
            state[i][j] = in_state[i][j] ^ roundkey[(round_number * Nb * 4) + (i * Nb) + j];
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
	/* Shift each column by coeficient, coeficent = nrow */
	/* No shift on 1st row */
	/* Shift 2nd row 1 byte */
	temp = state[0][1];
	state[0][1] = state[1][1];
	state[1][1] = state[2][1];
	state[2][1] = state[3][1];
	state[3][1] = temp;

	/* Shift 3rd row 2 bytes */
	temp = state[0][2];
	state[0][2] = state[2][2];
	state[2][2] = temp; 
	temp = state[1][2];
	state[1][2] = state[3][2];
	state[3][2] = temp; 

	/* Shift 4th row 3 bytes (same as 1 byte right shift) */
	temp = state[3][3];
	state[3][3] = state[2][3];
	state[2][3] = state[1][3];
	state[1][3] = state[0][3];
	state[0][3] = temp;
}

inline uint8_t xtime(uint8_t x)
{
  return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}

void AES_MixColumns(STATE_PARAM)
{
    uint8_t Tmp, Tm, t;
    for (int i = 0; i < 4; ++i)
    {
        t   = state[i][0];
        Tmp = state[i][0] ^ state[i][1] ^ state[i][2] ^ state[i][3] ;

        Tm  = state[i][0] ^ state[i][1];
        Tm = xtime(Tm);
        state[i][0] ^= Tm ^ Tmp;

        Tm  = state[i][1] ^ state[i][2];
        Tm = xtime(Tm);
        state[i][1] ^= Tm ^ Tmp;

        Tm  = state[i][2] ^ state[i][3];
        Tm = xtime(Tm);
        state[i][2] ^= Tm ^ Tmp;

        Tm  = state[i][3] ^ t;
        Tm = xtime(Tm);
        state[i][3] ^= Tm ^ Tmp;
    }
}

// FIXME there are more optimized ways of implementing this, e.g. using 32-bit variables for row lookup, and combining steps
// FIXME both x86 and ARM-v8 have specific instructions for implementing an AES round
void AES_encrypt_state(STATES_PARAM, NB_PARAM, SBOX_PARAM, ROUNDKEY_PARAM, unsigned int num_rounds) 
{
	AES_AddRoundKey(in_state, state, roundkey, Nb, 0); 

	for(unsigned int roundi=1; roundi<num_rounds; roundi++)
	{
		AES_SubBytes(state, sbox); 
		AES_ShiftRows(state);
		AES_MixColumns(state);
		AES_AddRoundKey(state, state, roundkey, Nb, roundi); 
	}
	//Last iteration without MixColumns
	AES_SubBytes(state, sbox);
	AES_ShiftRows(state); 
	AES_AddRoundKey(state, state, roundkey, Nb, num_rounds); 
}


/* Public functions */

void AES_encrypt(AES_data_t *AES_data, AES_time_t *t, int block)
{
    uint8_t *initial_state = AES_data->input_text+block*16;
    uint8_t *final_state = AES_data->encrypted_text+block*16;

    /* Operations per state */
    AES_encrypt_state((uint8_t (*)[4]) initial_state, (uint8_t (*)[4]) final_state, AES_data->key->Nb, AES_data->sbox, AES_data->expanded_key, AES_data->key->Nr);
}
