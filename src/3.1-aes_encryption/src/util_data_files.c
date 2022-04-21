/**
 * \file util_data_files.c 
 * \brief Benchmark #3.1 Data file load
 * \author Marc Sole Bonet (BSC)
 */


#include "util_data_files.h"



int load_data_from_files(

	uint8_t *input_plaintext,
	uint8_t *key,
	uint8_t *iv,

	unsigned int data_length,
	unsigned int key_size
	)
{
    frame8_t *input_frame = (frame8_t*) malloc(sizeof(frame8_t));
    frame8_t *key_frame = (frame8_t*) malloc(sizeof(frame8_t));
    frame8_t *iv_frame = (frame8_t*) malloc(sizeof(frame8_t));


    // create the key path
    char key_path[256];
    sprintf(key_path, "../../data/data_generation/datagen-3.1-aes_encryption/out/key-%d.bin", key_size);
    // init the offset map
    key_frame->w = key_size/8;
	key_frame->h = 1;
	key_frame->f = key;
    // read the binary file into the key frame
    if(!read_frame8(key_path, key_frame)) return FILE_LOADING_ERROR;

 
    // create the iv path
    char iv_path[256];
    sprintf(iv_path, "../../data/data_generation/datagen-3.1-aes_encryption/out/iv-%d.bin", data_length/16);
    // init the bad pixel map
    iv_frame->w = 16;
    iv_frame->h = 1;
    iv_frame->f = iv;
    // read the binary file into the iv frame
    if(!read_frame8(iv_path, iv_frame)) return FILE_LOADING_ERROR;

    // create the input frames path 
    char input_path[256];
    sprintf(input_path, "../../data/data_generation/datagen-3.1-aes_encryption/out/pt_%d.bin", data_length);
    // init the bad pixel map
    input_frame->w = data_length;
    input_frame->h = 1;
    input_frame->f = input_plaintext;
    // read the binary file into the iv frame
    if(!read_frame8(input_path, input_frame)) return FILE_LOADING_ERROR;

    return FILE_LOADING_SUCCESS;
}
