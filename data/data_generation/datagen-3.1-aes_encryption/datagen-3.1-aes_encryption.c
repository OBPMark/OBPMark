/**
 * \file datagen-3.1-encryption.c
 * \brief Data generation for Benchmark #3.1: AES encryption
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "obpmark_image.h"
#include "image_mem_util.h"
#include "image_file_util.h"

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1


//We use the frame8_t type to define the data
//The plaintext is in a frame of w = data_length and h = 1
//The initialization vector (iv) / counter nuance,  is stored in a frame of w = 16 and h = 1
//The key is defined as a frame8_t with a single frame of w = key_size/8 and h = 1

int arguments_handler(int argc, char ** argv, unsigned int *data_length, unsigned int *key_size, char *input_file, char *rand_key, char *key_file, char *rand_iv, char *iv_file);

void gen_key(frame8_t *key);
void gen_iv(frame8_t *iv);


void copy_frame(frame8_t *frame, frame8_t *base_frame);

/* Calibration data generation */

void gen_key(frame8_t *key)
{
	unsigned int i;
	for(i=0; i<key->w; i++)
    {
			 key->f[i] = (uint8_t)(rand() % 255);
	}
}

void gen_iv(frame8_t *iv)
{
	unsigned int i;
	for(i=0; i<iv->w; i++)
    {
			 iv->f[i] = (uint8_t)(rand() % 255);
	}
}

void copy_frame(
		frame8_t *frame,
		frame8_t *base_frame
	      )
{
	unsigned int x;
	for(x=0; x<frame->w; x++)
	{
		frame->f[x] = base_frame->f[x];	
	}
}


/* Generation of data set */

int benchmark3_1_write_files(
	frame8_t *key,
	frame8_t *iv,
	frame8_t *plaintext,
	unsigned int key_size,
	unsigned int data_length
	)
{
	unsigned int i;

	char key_name[50];
	char iv_name[50];
	char pt_name[50];

	unsigned int n_blocks = data_length / 16;

	sprintf(key_name,	 "out/key-%d.bin", 	key_size);
	sprintf(iv_name, 	 "out/iv-%d.bin", 	n_blocks);

	printf("Writing credentials data to files...\n");
	if(!write_frame8(key_name, key, 1)) {
		printf("error: failed to write key.\n");
		return 0;
	}

	if(!write_frame8(iv_name, iv, 1)) {
		printf("error: failed to write initialization vector / counter.\n");
		return 0;
	}

	/* Write plaintext data to files */
	printf("Writing plaintext data to files...\n");
    sprintf(pt_name, "out/pt_%d.bin", data_length);
    if(!write_frame8(pt_name, plaintext, 1)) {
        printf("error: failed to write plaintext data: %d\n", i);
        return 0;
	}
	return 1;
}

int benchmark3_1_data_gen(
	unsigned int data_length,
	unsigned int key_size,
	char *input_file,
	char rand_key,
	char *input_key,
	char rand_iv,
	char *input_iv
	)
{
    unsigned int i; 
    frame8_t key;
	frame8_t iv;
	frame8_t input_pt;
	frame8_t input_k;
	frame8_t input_i;

	frame8_t plaintext;

	/* Allocate calibration data buffers */
	printf("Allocating credentials data buffers...\n");
	if(!frame8_alloc(&input_pt, data_length, 1)) return 1; 
	if(!frame8_alloc(&key, key_size/8, 1)) return 1; 
	if(!frame8_alloc(&iv, 16, 1)) return 1; 

	/* Alloc frame buffers */
	printf("Allocating plaintext buffer...\n");
    if(!frame8_alloc(&plaintext, data_length, 1)) return 2; 

	/* Read input data */
	// read the binary file
	printf("Reading input data...\n");
	if(!read_frame8(input_file, &input_pt)) return 3;

	/* Generate credentials data (key and initialization vector)*/
	printf("Generating credentials...\n");
	if (rand_key) gen_key(&key);
    else{
        if(!frame8_alloc(&input_k, key_size/8, 1)) return 1; 
        if(!read_frame8(input_key, &input_k)) return 3;
        copy_frame(&key, &input_k);
    }
	if (rand_iv) gen_iv(&iv);
    else {
        if(!frame8_alloc(&input_i, 16, 1)) return 1; 
        if(!read_frame8(input_iv, &input_i)) return 3;
        copy_frame(&iv, &input_i);
    }

	/* Generate frame data */
	printf("Generating plaintext data...\n");
	copy_frame(&plaintext, &input_pt);

	/* Write encryption data to files */
	if(!benchmark3_1_write_files(&key, &iv, &plaintext, key_size, data_length)) {
		 /* Free buffers if error happen */
        frame8_free(&key);
        frame8_free(&iv);
        frame8_free(&plaintext);
		return 3;
	}

	/* Free buffers */
	frame8_free(&key);
	frame8_free(&iv);
    frame8_free(&plaintext);

	printf("Done.\n");

	return 0; 
}

/* Main */

int main(int argc, char *argv[])
{
	int ret;
	srand (28012015);
	/* Settings */
	unsigned int data_length, key_size;
	char input_file[100], key_file[100], iv_file[100];
	char rand_key = 1, rand_iv = 1;
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	int resolution = arguments_handler(argc,argv,&data_length,&key_size, input_file, &rand_key, key_file, &rand_iv, iv_file);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Data generation
	///////////////////////////////////////////////////////////////////////////////////////////////
	ret = benchmark3_1_data_gen(data_length, key_size, input_file, rand_key, key_file, rand_iv, iv_file);
	if(ret != 0) return ret;

	return 0;
}

void print_usage(const char * appName)
{
	printf("Usage: %s -l Size -k Size -i input_file [-h]\n", appName);
	printf(" -l: set length of the plaintext to be encrypted \n");
	printf(" -k: set key size (128|192|256) \n");
	printf(" -i: input plaintext \n");
	printf(" -y: input cipher key \n");
	printf(" -v: input initialization vector \n");
}

int arguments_handler(int argc, char ** argv, unsigned int *data_length, unsigned int *key_size, char *input_file, char *rand_key, char *key_file, char *rand_iv, char *iv_file){
	if (argc == 1){
		printf("-l, -k, -i need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	} 
	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
			case 'l' : args +=1; *data_length = atoi(argv[args]);break;
			case 'k' : args +=1; *key_size = atoi(argv[args]);break;
			case 'i' : args +=1; strcpy(input_file, argv[args]);break;
			case 'y' : args +=1; strcpy(key_file, argv[args]); *rand_key = 0; break;
			case 'v' : args +=1; strcpy(iv_file, argv[args]); *rand_iv = 0; break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	if (*data_length == 0 || *key_size == 0){
		printf("\n-l and -k need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (strcmp(input_file, "") == 0){
		printf("\n-i need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	if (*key_size != 128 && *key_size != 192 && *key_size != 256){
		printf("\n-k need to be set with 128, 192 or 256\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
	return OK_ARGUMENTS;			
}
