/**
 * \file datagen-1.2-radar.c
 * \brief Data generation for Benchmark #1.2: Radar
 * \author marc.solebonet@bsc.es
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


//We use the framefp_t type to define the data
//The plaintext is in a frame of w = data_length and h = 1
//The initialization vector (iv) / counter nuance,  is stored in a frame of w = 16 and h = 1
//The key is defined as a framefp_t with a single frame of w = key_size/8 and h = 1

int arguments_handler(int argc, char **argv, char *rand_data, unsigned int *abins, unsigned int *rbins, char *raw_file, char *ldr_file);

void gen_params(framefp_t *params);
void gen_iv(framefp_t *iv);


void copy_frame(framefp_t *frame, framefp_t *base_frame);

/* Calibration data generation */

void gen_params(framefp_t *params)
{
    float pi = (float) M_PI;
	PIXEL(params,0,0) = (float)rand()/(float)(RAND_MAX);                     //lambda [0,1]
    PIXEL(params,1,0) = 500.f + 1000.f * ((float)rand()/(float)(RAND_MAX));  //Pulse Repetition Frequency (PRF) [500, 1500]
    PIXEL(params,2,0) = (float)rand()/(float)(RAND_MAX);                     //tau [0,1]
    PIXEL(params,3,0) = 1e8 *(1 + 10*((float)rand()/(float)(RAND_MAX)));     //Sampling rate (fs) [1e8, 10e8]
    PIXEL(params,4,0) = 5000.f + 5000.f * ((float)rand()/(float)(RAND_MAX)); //Velocity [5000,10000]
    PIXEL(params,5,0) = 1e5 *(1 + 10*((float)rand()/(float)(RAND_MAX)));     //Ro [1e5, 10e5]
    PIXEL(params,6,0) = -pi/2 + pi*((float)rand()/(float)(RAND_MAX));        //Angle [-pi/2, pi/2]
    PIXEL(params,7,0) = 1e11 *(1 + 10*((float)rand()/(float)(RAND_MAX)));    //Ro [1e11, 10e11]
}

void gen_data(framefp_t *raw)
{
	unsigned int i;
	unsigned int j;
	for(i=0; i<raw->h; i++)
        for(j=0; j<raw->w; j++)
			 PIXEL(raw,i,j) = -15.5+31*((float)rand()/(float)(RAND_MAX));
}

void copy_frame(
		framefp_t *frame,
		framefp_t *base_frame
	      )
{
	unsigned int x;
	for(x=0; x<frame->w; x++)
	{
		frame->f[x] = base_frame->f[x];	
	}
}


/* Generation of data set */

int benchmark1_2_write_files(
	framefp_t *key,
	framefp_t *iv,
	framefp_t *plaintext,
	unsigned int key_size,
	unsigned int data_length
	)
{
	unsigned int i;

	char key_name[50];
	char iv_name[50];
	char pt_name[50];

	sprintf(key_name,	 "out/key-%d.bin", 	key_size);
	sprintf(iv_name, 	 "out/iv-%d.bin", key_size);

	printf("Writing credentials data to files...\n");
	if(!write_framefp(key_name, key, 1)) {
		printf("error: failed to write key.\n");
		return 0;
	}

	if(!write_framefp(iv_name, iv, 1)) {
		printf("error: failed to write initialization vector / counter.\n");
		return 0;
	}

	/* Write plaintext data to files */
	printf("Writing plaintext data to files...\n");
    sprintf(pt_name, "out/data_%d.bin", data_length);
    if(!write_framefp(pt_name, plaintext, 1)) {
        printf("error: failed to write plaintext data: %d\n", i);
        return 0;
	}
	return 1;
}

int benchmark1_2_data_gen(
    char *raw_file, 
    char *ldr_file, 
    char rand_data, 
    unsigned int abins, 
    unsigned int rbins
    )
{
    framefp_t raw;
	framefp_t params;
    
    int asize = abins;
    int rsize = rbins;

	/* Allocate calibration data buffers */
	printf("Allocating parameters buffer...\n");
	if(!framefp_alloc(&params, 8, 1)) return 1; 

	if (rand_data) gen_params(&params);
    else read_params(&params, ldr_file, &asize, &rsize);

	printf("Allocating data buffer...\n");
	if(!framefp_alloc(&params, asize, rsize*2)) return 1; 
    if(rand_data) gen_data()


    



	/* Alloc frame buffers */
	printf("Allocating plaintext buffer...\n");
    if(!framefp_alloc(&plaintext, data_length, 1)) return 2; 

	/* Read input data */
	// read the binary file
	printf("Reading input data...\n");
	if(!read_framefp(input_file, &input_pt)) return 3;

	/* Generate credentials data (key and initialization vector)*/
	printf("Generating credentials...\n");
	if (rand_key) gen_key(&key);
    else{
        if(!framefp_alloc(&input_k, key_size/8, 1)) return 1; 
        if(!read_framefp(input_key, &input_k)) return 3;
        copy_frame(&key, &input_k);
    }
	if (rand_iv) gen_iv(&iv);
    else {
        if(!framefp_alloc(&input_i, 16, 1)) return 1; 
        if(!read_framefp(input_iv, &input_i)) return 3;
        copy_frame(&iv, &input_i);
    }

	/* Generate frame data */
	printf("Generating plaintext data...\n");
	copy_frame(&plaintext, &input_pt);

	/* Write encryption data to files */
	if(!benchmark1_2_write_files(&key, &iv, &plaintext, key_size, data_length)) {
		 /* Free buffers if error happen */
        framefp_free(&key);
        framefp_free(&iv);
        framefp_free(&plaintext);
		return 3;
	}

	/* Free buffers */
	framefp_free(&key);
	framefp_free(&iv);
    framefp_free(&plaintext);

	printf("Done.\n");

	return 0; 
}

/* Main */

int main(int argc, char *argv[])
{
	int ret;
	srand (28012015);
	/* Settings */
	unsigned int abins = 0, rbins = 0; //size of azimuth data and range data
	char raw_file[100], ldr_file[100];
	char rand_data = 0;
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Arguments  
	///////////////////////////////////////////////////////////////////////////////////////////////
	int resolution = arguments_handler(argc, argv, &rand_data, &abins, &rbins, input_file, ldr_file);
	if (resolution == ERROR_ARGUMENTS){
		exit(-1);
	}
	///////////////////////////////////////////////////////////////////////////////////////////////
	// Data generation
	///////////////////////////////////////////////////////////////////////////////////////////////
	ret = benchmark1_2_data_gen(raw_file, ldr_file, rand_data, abins, rbins);
	if(ret != 0) return ret;

	return 0;
}

void print_usage(const char * appName)
{
	printf("Usage: %s -i raw_file -p param_file [-h]\n", appName);
	printf("Usage: %s -r -l Size -w Size [-h]\n", appName);
	printf(" -i: input raw data file \n");
	printf(" -p: input parameters (ldr) file \n");
	printf(" -r: use random data (discards -i and -p) \n");
	printf(" -l: input length (if random data)\n");
	printf(" -w: input width (if random data) \n");
}

int arguments_handler(int argc, char ** argv, char *rand_data, unsigned int *abins, unsigned int *rbins, char *raw_file, char *ldr_file){
    if (argc == 1){
        printf("-i, -p or -r, -l, -w need to be set\n\n");
        print_usage(argv[0]);
        return ERROR_ARGUMENTS;
    } 
    for(unsigned int args = 1; args < argc; ++args)
    {
        switch (argv[args][1]) {
            case 'i' : args +=1; strcpy(raw_file, argv[args]);break;
            case 'p' : args +=1; strcpy(ldr_file, argv[args]);break;
            case 'l' : args +=1; *abins = atoi(argv[args]);break;
            case 'w' : args +=1; *rbins = atoi(argv[args]);break;
            case 'r' : *rand_data = 1;break;
            case 'i' : args +=1; strcpy(input_file, argv[args]);break;
            case 'y' : args +=1; strcpy(key_file, argv[args]); *rand_key = 0; break;
            case 'v' : args +=1; strcpy(iv_file, argv[args]); *rand_iv = 0; break;
            default: print_usage(argv[0]); return ERROR_ARGUMENTS;
        }

    }
    if (*rand_data){
        if (*abins == 0 || *rbins == 0){
            printf("\nif random, -l and -w need to be set\n\n");
            print_usage(argv[0]);
            return ERROR_ARGUMENTS;
        }
    }
    else{
        if (strcmp(raw_file, "") == 0){
            printf("\n-i need to be set\n\n");
            print_usage(argv[0]);
            return ERROR_ARGUMENTS;
        }
        if (strcmp(ldr_file, "") == 0){
            printf("\n-p need to be set\n\n");
            print_usage(argv[0]);
            return ERROR_ARGUMENTS;
        }
    }
    return OK_ARGUMENTS;			
}
