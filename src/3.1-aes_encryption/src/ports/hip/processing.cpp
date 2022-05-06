/**
 * \file processing.cu
 * \author Marc Sole Bonet (BSC)
 * \brief AES encryption implementation for OBPMark. CUDA implementation
 */
#include "processing.h"
#include <hip/hip_profile.h>

#include <stddef.h>
#include <stdint.h> 
#include <stdio.h>
#include <stdlib.h>


/* Internal function declarations */

__device__ void AES_AddRoundKey(STATES_PARAM, ROUNDKEY_PARAM, NB_PARAM, unsigned int round_number);
__device__ void AES_SubBytes(STATE_PARAM, SBOX_PARAM);
__device__ void AES_ShiftRows(STATE_PARAM);
__device__ void AES_MixColumns(STATE_PARAM);
__device__ void AES_encrypt_state(STATES_PARAM, NB_PARAM, SBOX_PARAM, ROUNDKEY_PARAM, unsigned int num_rounds);


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

#define AES_RotWord(n) ((n>>8) | (n<<24))

__device__ void AES_SubWord(uint32_t word, SBOX_PARAM, uint32_t *result)
{
    *result = sbox[word>>24]<<24|sbox[(uint8_t)(word>>16)]<<16|sbox[(uint8_t)(word>>8)]<<8|sbox[(uint8_t)word];
}

__global__ void AES_KeyExpansion(DATA_PARAM)
{
    uint32_t temp;
    uint32_t *expanded_key = (uint32_t*) AES_data->expanded_key;
    int Nk = AES_data->key->Nk, Nr = AES_data->key->Nr, Nb = AES_data->key->Nb;
    expanded_key[threadIdx.x] = *((uint32_t *)&AES_data->key->value[4*threadIdx.x]);
    __syncthreads();
    if(threadIdx.x == 0)
        for(int i = Nk; i < Nb * (Nr+1); i++) {
            temp = expanded_key[i-1];
            if (i%Nk == 0) {
                AES_SubWord(AES_RotWord(temp), AES_data->sbox, &temp);
                temp ^= AES_data->rcon[i/Nk];
            }
            else if (Nk > 6 && i%Nk == 4){
                AES_SubWord(temp,AES_data->sbox,&temp);
            }
            expanded_key[i] = expanded_key[i-Nk] ^ temp;
        }
}

__device__ void AES_AddRoundKey(STATES_PARAM, ROUNDKEY_PARAM, NB_PARAM,  unsigned int round_number)
{
#ifdef CUDA_FINE
	int i = threadIdx.y;
	int j = threadIdx.z;
    state[i][j] = in_state[i][j] ^ roundkey[(round_number * Nb * 4) + (i * Nb) + j];
#else
	for(int i=0; i<Nb; i++) {
		for(int j=0; j<4; j++) {
            state[i][j] = in_state[i][j] ^ roundkey[(round_number * Nb * 4) + (i * Nb) + j];
		}
	}
#endif
}

__device__ void AES_SubBytes(STATE_PARAM, uint8_t *sbox)
{
#ifdef CUDA_FINE
	int i = threadIdx.y;
	int j = threadIdx.z;
    state[i][j] = sbox[state[i][j]]; 
#else
	for(int i=0; i<4; i++) {
		for(int j=0; j<4; j++) {
			state[i][j] = sbox[state[i][j]]; 
		}
	}
#endif
}

__device__ void AES_ShiftRows(STATE_PARAM)
{
#ifdef CUDA_FINE
	int i = threadIdx.y;
	int j = threadIdx.z;
	uint8_t temp= state[(i+j)%4][j];
    __syncthreads();
	state[i][j] = temp;
#else
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
#endif
}

#define xtime(x) ((x<<1) ^ (((x>>7) & 1) * 0x1b))

__device__ void AES_MixColumns(STATE_PARAM)
{
#ifdef CUDA_FINE
    int i = threadIdx.y;
    int j = threadIdx.z;
    uint8_t Tmp, Tm;
    Tmp = state[i][0] ^ state[i][1] ^ state[i][2] ^ state[i][3] ;
    Tm = state[i][j] ^ state[i][(j+1)%4];
    Tm = xtime(Tm);
    __syncthreads();
    state[i][j] ^= Tm ^ Tmp;
#else
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
#endif
}

#ifdef CUDA_FINE
#define SYNC __syncthreads()
#else 
#define SYNC 
#endif
__device__ void AES_encrypt_state(STATES_PARAM, NB_PARAM, SBOX_PARAM, ROUNDKEY_PARAM, unsigned int num_rounds) 
{
    AES_AddRoundKey(in_state, state, roundkey, Nb, 0); 
    SYNC;
    for(unsigned int roundi=1; roundi<num_rounds; roundi++)
    {
        AES_SubBytes(state, sbox); 
        SYNC;
        AES_ShiftRows(state);
        SYNC;
        AES_MixColumns(state);
        SYNC;
        AES_AddRoundKey(state, state, roundkey, Nb, roundi); 
        SYNC;
    }
    //Last iteration without MixColumns
    AES_SubBytes(state, sbox);
    SYNC;
    AES_ShiftRows(state); 
    SYNC;
    AES_AddRoundKey(state, state, roundkey, Nb, num_rounds); 
    SYNC;
}

__device__ void counter_add(uint8_t *iv, uint64_t block, int id){
    uint64_t carry;
    uint8_t *counter = iv+16*(blockIdx.x*blockDim.x+threadIdx.x);
    carry = iv[id] + block;
    if (block <=(255-iv[id]) || id == 0) {
        counter[id] = carry;
        for(int i = id-1; i>=0; i--)  counter[i] = iv[i];
        return;
    }
    else {
        counter[id] = carry;
        carry >>= 8;
        counter_add(iv, carry, id-1);
    }
}
__device__ void counter_add(uint8_t *iv, uint64_t block){
#ifdef CUDA_FINE
    if (threadIdx.y != 0 || threadIdx.z != 0) return;
#endif
    counter_add(iv, block, 15);
}


/* Public functions */


__global__ void AES_encrypt(AES_values_t *AES_data)
{
    int block = blockIdx.x;
    int thread = threadIdx.x;
    int offset =  16*(thread+block*blockDim.x);
    if (offset >= AES_data->data_length) return;
    uint8_t *plaintext = AES_data->plaintext+offset;
    uint8_t *counter = AES_data->iv+offset;
    uint8_t *final_state = AES_data->cyphertext+offset;

    /*set the counter value */
    counter_add(AES_data->iv, offset>>4);
    SYNC;

    /* Operations per state */
    AES_encrypt_state((uint8_t (*)[4]) counter, (uint8_t (*)[4]) final_state, AES_data->key->Nb, AES_data->sbox, AES_data->expanded_key, AES_data->key->Nr);

    /* XOR iv with plaintext */
#ifdef CUDA_FINE
     final_state[4*threadIdx.y+threadIdx.z] ^= plaintext[4*threadIdx.y+threadIdx.z];
#else 
    for(int y = 0; y < AES_data->key->Nb; y++) *((uint32_t*) &final_state[4*y]) ^= *((uint32_t*) &plaintext[4*y]);
#endif
}
