#htvar kernel_code

#define STATE_PARAM	global unsigned char *state
#define INSTATE_PARAM global unsigned char *in_state
#define state(x,y) state[x*4+y]
#define in_state(x,y) in_state[x*4+y]
#define STATES_PARAM INSTATE_PARAM, STATE_PARAM
#define ROUNDKEY_PARAM	global  unsigned char *roundkey
#define SBOX_PARAM global const unsigned char *sbox
#define RCON_PARAM global unsigned char *rcon
#define NB_PARAM const unsigned int Nb
#define NR_PARAM const unsigned int Nr
#define NK_PARAM const unsigned int Nk
#define KEY_VALUE global const unsigned char *value

#define KEY_PARAM NB_PARAM, NR_PARAM, NK_PARAM, KEY_VALUE

#define AES_ECB 0
#define AES_CTR 1
#define MODE_PARAM const unsigned int mode

#define DATA_PARAM global unsigned char *plaintext, global unsigned char *cyphertext, global unsigned char *iv, NB_PARAM, NR_PARAM, SBOX_PARAM, ROUNDKEY_PARAM, MODE_PARAM

void printState(STATE_PARAM, int Nb){
    for(int i =0; i<Nb; i++){
        for(int j =0; j<4; j++)
            printf("%#x ", state(j,i));
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

unsigned int AES_RotWord(unsigned int n)
{
    return (n>>8) | (n<<24);
}

unsigned int AES_SubWord(unsigned int word, SBOX_PARAM)
{
    return sbox[word>>24]<<24|sbox[(unsigned char)(word>>16)]<<16|sbox[(unsigned char)(word>>8)]<<8|sbox[(unsigned char)word];
}

void kernel
AES_KeyExpansion(KEY_PARAM, ROUNDKEY_PARAM, SBOX_PARAM, RCON_PARAM)
{
	unsigned char temp[4];
	unsigned char single_temp;
    int i = 0;
    while(i < Nk){
        roundkey[i*4+0] = value[i*4+0];
        roundkey[i*4+1] = value[i*4+1];
        roundkey[i*4+2] = value[i*4+2];
        roundkey[i*4+3] = value[i*4+3];
        i++;
    }
    for(; i < Nb * (Nr+1); i++) {
        temp[0] = roundkey[(i-1)*4];
        temp[1] = roundkey[1+(i-1)*4];
        temp[2] = roundkey[2+(i-1)*4];
        temp[3] = roundkey[3+(i-1)*4];
        if (i%Nk == 0) {
            single_temp = temp[0];
            temp[0] = sbox[temp[1]] ^ rcon[i/Nk];
            temp[1] = sbox[temp[2]];
            temp[2] = sbox[temp[3]];
            temp[3] = sbox[single_temp];
        }
        else if (Nk > 6 && i%Nk == 4){
            temp[0] = sbox[temp[0]];
            temp[1] = sbox[temp[1]];
            temp[2] = sbox[temp[2]];
            temp[3] = sbox[temp[3]];
        }
        roundkey[i*4+0] = roundkey[(i-Nk)*4+0] ^ temp[0];
        roundkey[i*4+1] = roundkey[(i-Nk)*4+1] ^ temp[1];
        roundkey[i*4+2] = roundkey[(i-Nk)*4+2] ^ temp[2];
        roundkey[i*4+3] = roundkey[(i-Nk)*4+3] ^ temp[3];
    }
}

void AES_AddRoundKey(STATES_PARAM, ROUNDKEY_PARAM, NB_PARAM,  unsigned int round_number)
{
for(int i=0; i<Nb; i++) {
    for(int j=0; j<4; j++) {
        state(i,j) = in_state(i,j) ^ roundkey[(round_number * Nb * 4) + (i * Nb) + j];
    }
}
}

void AES_SubBytes(STATE_PARAM, SBOX_PARAM)
{
for(int i=0; i<4; i++) {
    for(int j=0; j<4; j++) {
        state(i,j) = sbox[state(i,j)]; 
    }
}
}

void AES_ShiftRows(STATE_PARAM)
{
unsigned char temp;
/* Shift each column by coeficient, coeficent = nrow */
/* No shift on 1st row */
/* Shift 2nd row 1 byte */
temp = state(0,1);
state(0,1) = state(1,1);
state(1,1) = state(2,1);
state(2,1) = state(3,1);
state(3,1) = temp;

/* Shift 3rd row 2 bytes */
temp = state(0,2);
state(0,2) = state(2,2);
state(2,2) = temp; 
temp = state(1,2);
state(1,2) = state(3,2);
state(3,2) = temp; 

/* Shift 4th row 3 bytes (same as 1 byte right shift) */
temp = state(3,3);
state(3,3) = state(2,3);
state(2,3) = state(1,3);
state(1,3) = state(0,3);
state(0,3) = temp;
}

unsigned char xtime(unsigned char x)
{
return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
}

void AES_MixColumns(STATE_PARAM)
{
unsigned char Tmp, Tm, t;
for (int i = 0; i < 4; ++i)
{
    t   = state(i,0);
    Tmp = state(i,0) ^ state(i,1) ^ state(i,2) ^ state(i,3) ;

    Tm  = state(i,0) ^ state(i,1);
    Tm = xtime(Tm);
    state(i,0) ^= Tm ^ Tmp;

    Tm  = state(i,1) ^ state(i,2);
        Tm = xtime(Tm);
        state(i,1) ^= Tm ^ Tmp;

        Tm  = state(i,2) ^ state(i,3);
        Tm = xtime(Tm);
        state(i,2) ^= Tm ^ Tmp;

        Tm  = state(i,3) ^ t;
        Tm = xtime(Tm);
        state(i,3) ^= Tm ^ Tmp;
    }
}

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

void counter_add_rec(global unsigned char *iv, unsigned int block, int id){
    unsigned int carry;
    carry = iv[id] + block;
    if (block <=(255-iv[id]) || id == 0) {
        iv[id] = carry;
        return;
    }
    else {
        iv[id] = carry;
        carry >>= 8;
        counter_add_rec(iv, carry, id-1);
    }
}
void counter_add(global unsigned char *iv, unsigned int block){
    counter_add_rec(iv, block, 15);
}

void kernel
AES_encrypt(DATA_PARAM)
{
    global unsigned char *initial_state = plaintext+get_global_id(0)*16;
    global unsigned char *final_state = cyphertext+get_global_id(0)*16;
    global unsigned char *counter = iv+get_global_id(0)*16;
    switch(mode){
        case AES_ECB:
            /* Operations per state */
            AES_encrypt_state(initial_state, final_state, Nb, sbox, roundkey, Nr);
            break;

        case AES_CTR:
            /* set the counter value */
            counter_add(counter, get_global_id(0));

            /* Operations per state */
            AES_encrypt_state(counter, final_state, Nb, sbox, roundkey, Nr);

            /* XOR iv with plaintext */
            for(int y = 0; y < Nb; y++) *((unsigned int*) &final_state[4*y]) ^= *((unsigned int*) &initial_state[4*y]);
            break;
    }
}

#htendvar
