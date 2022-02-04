/**
 * \file device.c
 * \brief Benchmark #3.1 CPU version (sequential) device initialization. 
 * \author Marc Sole(BSC)
 */
#include "device.h"
#include "processing.h"

void init(
	AES_data_t *AES_data,
	AES_time_t *t,
	char *device_name
	)
{
    init(AES_data,t, 0,0, device_name);
}

void init(
	AES_data_t *AES_data,
	AES_time_t *t,
	int platform,
	int device,
	char *device_name
	)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
	//Time object init for multiple

}


bool device_memory_init(
	AES_data_t *AES_data,
    unsigned int key_size,
    unsigned int data_size
	)
{	
	/* key configuration values initialization */
    AES_data->key = (AES_key_t*) malloc(sizeof(AES_key_t));
	AES_data->key->size = (AES_keysize_t) key_size;
	switch(key_size) {
		case AES_KEY128: AES_data->key->Nk = 4; AES_data->key->Nr = 10; break;
		case AES_KEY192: AES_data->key->Nk = 6; AES_data->key->Nr = 12; break;
		case AES_KEY256: AES_data->key->Nk = 8; AES_data->key->Nr = 14; break;
	}
	AES_data->key->Nb = 4;
	/* key value memory allocation */
	AES_data->key->value = (uint8_t*)malloc(sizeof(uint8_t)*key_size);

	/* memory allocation for input and output texts */
    AES_data->input_text = (uint8_t*) malloc(sizeof(uint8_t)*data_size);
    AES_data->encrypted_text = (uint8_t*) malloc(sizeof(uint8_t)*data_size);
    AES_data->data_length = data_size;

    /* allocate constant lookup tables */
    /* memory for sbox (256 uint8) */
    AES_data->sbox = (uint8_t*) malloc(sizeof(uint8_t)*256);
    /* memory for rcon (11 uint8) */
    AES_data->rcon = (uint8_t*) malloc(sizeof(uint8_t)*11);

    /* memory for roundkey (expanded key Nb*(Nr+1) uint32) */
    AES_data->expanded_key = (uint8_t*) malloc(sizeof(uint32_t)*AES_data->key->Nb*(AES_data->key->Nr+1));

    return true;
}

void copy_memory_to_device(
	AES_data_t *AES_data,
	uint8_t *input_key,
	uint8_t *input_text,
	uint8_t *input_sbox,
	uint8_t *input_rcon
	)
{
    /* initialize key value */
    memcpy(AES_data->key->value, input_key, sizeof(uint8_t)*AES_data->key->size/8);
    /* initialize input text */
    memcpy(AES_data->input_text, input_text, sizeof(uint8_t)*AES_data->data_length);
    /* initialize sbox */
    memcpy(AES_data->sbox, input_sbox, sizeof(uint8_t)*256);
    /* initialize rcon */
    memcpy(AES_data->rcon, input_rcon, sizeof(uint8_t)*11);
}


void process_benchmark(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{    
    int n_blocks = AES_data->data_length/ (4*AES_data->key->Nb);
    T_START(t->t_test);
    T_START_VERBOSE(t->t_key_expand);
    AES_KeyExpansion(AES_data->key, (uint32_t*) AES_data->expanded_key, AES_data->sbox, AES_data->rcon);
    T_STOP_VERBOSE(t->t_key_expand);
    T_START_VERBOSE(t->t_encrypt);
    //TODO time each encrytp block
    for(int b = 0; b < n_blocks; b++){
        AES_encrypt(AES_data, t, b);
    }
    T_STOP_VERBOSE(t->t_encrypt);
    T_STOP(t->t_test);
}

void copy_memory_to_host(
	AES_data_t *AES_data,
	uint8_t *output
	)
{
    memcpy(output, AES_data->encrypted_text, sizeof(uint8_t)*AES_data->data_length);
}


float get_elapsed_time(
//	image_data_t *image_data, 
	AES_time_t *t, 
	bool csv_format
	)
{
    // FIXME with new time format
	//float elapsed =  (device_object->end.tv_sec - device_object->start.tv_sec) * 1000 + (device_object->end.tv_nsec - device_object->start.tv_nsec) / 1000000;
	float elapsed = T_TO_SEC(t->t_test)*1000;
    if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (float) 0, elapsed, (float) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (float) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", elapsed);
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (float) 0);
    }
	return elapsed;
}


void clean(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{
	unsigned int frame_i;

	/* Clean time */
	free(t);

	free(AES_data->key->value);
	free(AES_data->key);
	free(AES_data->input_text);
	free(AES_data->encrypted_text);
	free(AES_data->sbox);
	free(AES_data);
}
