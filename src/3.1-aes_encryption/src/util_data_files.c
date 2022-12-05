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
	unsigned int key_size,
	char *input_folder
	)
{
    frame8_t *input_frame = (frame8_t*) malloc(sizeof(frame8_t));
    frame8_t *key_frame = (frame8_t*) malloc(sizeof(frame8_t));
    frame8_t *iv_frame = (frame8_t*) malloc(sizeof(frame8_t));

    bool default_input_folder = false;

    if (strcmp(input_folder,"") == 0)
    {
        default_input_folder = true;
    }

    // create the key path
    char key_path[256];
    if(default_input_folder)
    {
       sprintf(key_path, "%s/%d/3.1-aes_encryption-key_%d.bin", DEFAULT_INPUT_FOLDER,key_size, key_size);
    }
    else 
    {
       sprintf(key_path, "%s/3.1-aes_encryption-key_%d.bin", input_folder, key_size);
    }
    // init the offset map
    key_frame->w = key_size/8;
	key_frame->h = 1;
	key_frame->f = key;
    // read the binary file into the key frame
    if(!read_frame8(key_path, key_frame)) return FILE_LOADING_ERROR;

 
    // create the iv path
    char iv_path[256];
    if(default_input_folder)
    {
       sprintf(iv_path, "%s/%d/3.1-aes_encryption-iv_%d.bin", DEFAULT_INPUT_FOLDER, key_size, key_size);
    }
    else 
    {
       sprintf(iv_path, "%s/3.1-aes_encryption-iv_%d.bin", input_folder, key_size);
    }
    // init the bad pixel map
    iv_frame->w = 16;
    iv_frame->h = 1;
    iv_frame->f = iv;
    // read the binary file into the iv frame
    if(!read_frame8(iv_path, iv_frame)) return FILE_LOADING_ERROR;

    // create the input frames path 
    char input_path[256];
    if(default_input_folder)
    {
       sprintf(input_path, "%s/%d/3.1-aes_encryption-data_%d.bin", DEFAULT_INPUT_FOLDER,key_size, data_length);
    }
    else 
    {
       sprintf(input_path, "%s/3.1-aes_encryption-data_%d.bin", input_folder, data_length);
    }
    // init the bad pixel map
    input_frame->w = data_length;
    input_frame->h = 1;
    input_frame->f = input_plaintext;
    // read the binary file into the iv frame
    if(!read_frame8(input_path, input_frame)) return FILE_LOADING_ERROR;

    return FILE_LOADING_SUCCESS;
}
