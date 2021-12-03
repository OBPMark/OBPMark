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

#define STATE_PARAM	uint8_t state[4][4]
#define ROUNDKEY_PARAM	uint8_t *roundkey


/* Internal function declarations */

void AES_KeyExpansion(AES_key_t *key, uint32_t *expanded_key, uint8_t *sbox);
void AES_AddRoundKey(STATE_PARAM, ROUNDKEY_PARAM, unsigned int Nb, unsigned int round_number);
void AES_SubBytes(STATE_PARAM, uint8_t *sbox);
void AES_ShiftRows(STATE_PARAM);
void AES_MixColumns(STATE_PARAM);
void AES_encrypt_state(STATE_PARAM, uint8_t *sbox, ROUNDKEY_PARAM, unsigned int num_rounds);


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

inline uint32_t RotWord(uint32_t n)
{
    return (n>>8) | (n<<24);
}

uint32_t SubWord(uint32_t word, uint8_t *sbox)
{
    return sbox[word>>24]<<24|sbox[(uint8_t)(word>>16)]<<16|sbox[(uint8_t)(word>>8)]<<8|sbox[(uint8_t)word];
}

void AES_KeyExpansion(AES_key_t *key, uint32_t *expanded_key, uint8_t *sbox, uint8_t *rcon)
{
	uint32_t temp;
	int Nk = key->Nk, Nr = key->Nr, Nb = key->Nb;

    int i = 0;
    while(i < Nk){
        expanded_key[i] = *((uint32_t *)&key->value[4*i]);
        i++;
    }

    for(; i < Nb * (Nr+1); i++) {
//        printf("i: %d\t", i);
        temp = expanded_key[i-1];
//        printf("temp: %#010x\t", temp);
        if (i%Nk == 0) {
//            temp = RotWord(temp);
//            printf("after rotword: %#010x\t", temp);
//            temp = SubWord(temp, sbox);
//            printf("after SubWord: %#010x\t", temp);
//            printf("Rcon: %#010x\t", Rcon[i/Nk]);
//            temp ^= Rcon[i/Nk];
//            printf("after xor rcon: %#010x\t", temp);
            temp = SubWord(RotWord(temp), sbox) ^ rcon[i/Nk];

        }
        else if (Nk > 6 && i%Nk == 4){
            temp = SubWord(temp,sbox);
//            printf("after SubWord: %#010x\t", temp);
        }

//        printf("w[i-nk]: %#x\t", expanded_key[i-Nk]);
        expanded_key[i] = expanded_key[i-Nk] ^ temp;
//        uint32_t b0,b1,b2,b3;
//        b0 = (expanded_key[i] & 0x000000ff) << 24u;
//        b1 = (expanded_key[i] & 0x0000ff00) << 8u;
//        b2 = (expanded_key[i] & 0x00ff0000) >> 8u;
//        b3 = (expanded_key[i] & 0xff000000) >> 24u;
//
//        //printf("%#x\n", b0 | b1 | b2 | b3);
    }
}

void AES_AddRoundKey(STATE_PARAM, ROUNDKEY_PARAM, unsigned int Nb, unsigned int round_number)
{
	for(int i=0; i<Nb; i++) {
		for(int j=0; j<4; j++) {
			state[i][j] = state[i][j] ^ roundkey[(round_number * Nb * 4) + (i * Nb) + j];
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

void AES_encrypt_state(STATE_PARAM, unsigned int Nb, uint8_t *sbox, ROUNDKEY_PARAM, unsigned int num_rounds) 
{
#ifdef DEBUG
	printf("Round %d\n",0);
	printf("input\n");
	printState(state);
	printf("key\n");
	printKey(roundkey, 0);
#endif
	AES_AddRoundKey(state, roundkey, Nb, 0); 

	/* 3. Rounds */
	for(unsigned int roundi=1; roundi<num_rounds; roundi++)
	{
#ifdef DEBUG
	printf("Round %d\n",roundi);
	printf("starting\n");
	printState(state);
#endif
		/* 1. SubBytes 
		 * 2. ShiftRows 
		 * 3. MixColumns 
		 * 4. AddRoundKey */
		AES_SubBytes(state, sbox); 
#ifdef DEBUG
	printf("Round %d\n",roundi);
        printf("after_subbytes:\n");
        printState(state);
#endif
		AES_ShiftRows(state);
#ifdef DEBUG
        printf("after_shiftrows:\n");
        printState(state);
#endif
		AES_MixColumns(state);
#ifdef DEBUG
        printf("after_mixcolumns:\n");
        printState(state);
        printf("key\n");
        printKey(roundkey, roundi);
#endif
		AES_AddRoundKey(state, roundkey, Nb, roundi); 
	}

	/* Final round 
	 * 	- 1. SubBytes
	 * 	- 2. ShiftRows
	 * 	- 3. AddRoundKey */
#ifdef DEBUG
	printf("starting\n");
	printState(state);
#endif
	AES_SubBytes(state, sbox);
#ifdef DEBUG
        printf("after_subbytes:\n");
        printState(state);
#endif
	AES_ShiftRows(state); 
#ifdef DEBUG
        printf("after_shiftrows:\n");
        printState(state);
        printf("key\n");
        printKey(roundkey, num_rounds);
#endif
	AES_AddRoundKey(state, roundkey, Nb, num_rounds); 
}


/* Public functions */

// FIXME there are more optimized ways of implementing this, e.g. using 32-bit variables for row lookup, and combining steps
// FIXME both x86 and ARM-v8 have specific instructions for implementing an AES round
void AES_encrypt(AES_data_t *AES_data)
{
    uint8_t *state = AES_data->input_text;

    AES_KeyExpansion(AES_data->key, (uint32_t*) AES_data->expanded_key, AES_data->sbox, AES_data->rcon);
    

//	while(1) // FIXME exit criteria 
//	{
		/* Extract data from buffer to state */
		// FIXME extract data from buf into state
//		for(int i=0; i<4; i++) {
//			for(int j=0; j<4; j++) {
//				state[i][j] = data[x]; // FIXME what about end of file? not multiple of 16?
//				x++;
//			}
//		}

		/* Operations per state */
		AES_encrypt_state((uint8_t (*)[4]) state, AES_data->key->Nb, AES_data->sbox, AES_data->expanded_key, AES_data->key->Nr);
#ifdef DEBUG
		puts("final:");
		printState((uint8_t (*)[4]) state, AES_data->key->Nb);
#endif

		/* Save output */
		int y = 0;
		for(int i=0; i<4; i++) {
			for(int j=0; j<4; j++) {
				AES_data->encrypted_text[y] = ((uint8_t (*)[4])state)[i][j]; // FIXME what about end of file? not multiple of 16?
				y++;
			}
		}
	//}
}
