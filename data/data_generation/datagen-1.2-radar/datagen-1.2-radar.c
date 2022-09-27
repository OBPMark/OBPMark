/**
 * \file datagen-1.2-radar.c
 * \brief Data generation for Benchmark #1.2: Radar
 * \author marc.solebonet@bsc.es
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "obpmark_image.h"
#include "image_mem_util.h"
#include "image_file_util.h"

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

typedef struct {
    float lambda;
    float PRF;
    float tau;
    float fs;
    float vr;
    float ro;
    float slope;
    uint32_t asize;     //total number of azimuth samples
    uint32_t avalid;    //number of meaningfull azimuth samples in a patch
    uint32_t apatch;    //total number of azimuth samples in a patch
    uint32_t rsize;     //total number of range samples = samples in a patch
    uint32_t rvalid;
    uint32_t npatch;    //number of patches in the image
}radar_params_t;

int arguments_handler(int argc, char **argv, char *rand_data, unsigned int *nlines, unsigned int *nsamples, char *raw_file, char *ldr_file);
int write_params(char filename[], radar_params_t *array);

unsigned int pow2(unsigned int a) { return pow(2,a); }

/* Calibration data generation */

void gen_params(radar_params_t *params, uint32_t height, uint32_t width)
{
    printf("Generating parameter data...\n");
    float pi = (float) M_PI;
    params->lambda = (float)rand()/(float)(RAND_MAX)*0.2;                 //lambda [0,0.2]
    params->PRF    = 500.f + 1000.f * ((float)rand()/(float)(RAND_MAX));  //Pulse Repetition Frequency (PRF) [500, 1500]
    params->tau    = 1e-4 * (float)rand()/(float)(RAND_MAX);                     //tau [0,1e-4]
    params->fs     = 1e7 *(1 + 10*((float)rand()/(float)(RAND_MAX)));     //Sampling rate (fs) [1e7, 10e7]
    params->vr     = 5000.f + 5000.f * ((float)rand()/(float)(RAND_MAX)); //Velocity [5000,10000]
    params->ro     = 1e5 *(1 + 10*((float)rand()/(float)(RAND_MAX)));     //Ro [1e5, 10e5]
    params->slope  = 1e11 *(1 + 10*((float)rand()/(float)(RAND_MAX)));    //Slope [1e11, 10e11]
    params->rsize  = width;                                               //Number of valid range points
    params->rvalid = width - round(params->tau*params->fs);
    if(params->rvalid < (width/2)) params->rvalid = width;
    params->asize  = height;                                              //Total azimuth samples
    float dopp = 150.f + 100.f * ((float)rand()/(float)(RAND_MAX));       //Random doppler freq
    float c = 2.99792458e8;                                               //Speed of light
    float rnge = params->ro+(width/2)*(c/(2*params->fs));                 //range perpendicular to azimuth
    float rdc = rnge/sqrt(1-pow(params->lambda*dopp/(2*params->vr),2));   //squinted range
    float tauz = (rdc*(params->lambda/10) * 0.8) / params->vr;            //Tau in the azimuth
    if(height < 1024) params->avalid = height;
    else params->avalid  = 2048 - ceil(tauz*params->PRF);                 //Number of valid azimuth points
    params->apatch = pow(2, ceil(log2(params->avalid + 1)));              //Number of azimuth samples per patch
    params->npatch = height/params->avalid;                               //Number of patches
}

void gen_data(float *raw, float height, float width)
{
    printf("Generating input data...\n");
    unsigned int i, I = height * width * 2; //times two for complex
    for(i=0; i<I; i++)
        raw[i] = -15.5+31*((float)rand()/(float)(RAND_MAX));
}

/* Generation of data set */

int benchmark1_2_write_files(
        framefp_t *patches,
        radar_params_t *params
        )
{
    char data_name[100];
    char para_name[50];
    char dir_name[25];

    sprintf(dir_name, "out/%d_%d", params->asize, params->rsize);
    //create directory
    struct stat st = {0};
    if (stat("out", &st) == -1){
        mkdir("out", 0755);
    }
    if (stat(dir_name, &st) == -1){
        printf("Creating directory...\n");
        mkdir(dir_name, 0755);
    }
    sprintf(para_name, "%s/1.2-radar-params_%d_%d.bin", dir_name, params->asize, params->rsize);

    printf("Writing parameter data to file...\n");
    if(!write_params(para_name, params)) {
        printf("error: failed to write parameters.\n");
        return 0;
    }

    /* Write input data to files */
	printf("Writing patch data to files...\n");
	for(int i=0; i<params->npatch; i++)
	{
		sprintf(data_name, "%s/1.2-radar-patch_%d_%d_%d.bin", dir_name, i, params->apatch, params->rsize);
		if(!write_framefp(data_name, &patches[i], 1)) {
			printf("error: failed to write frame data: %d\n", i);
			return 0;
		}
	}
    return 1;
}

int read_raw(char filename[], float *array, unsigned int height, unsigned int width)
{
    printf("Reading raw data...\n");
    FILE *framefile;
    static const uint8_t data_width = sizeof(float);
    uint32_t nelm = height * width * 2;
    size_t bytes_read;
    size_t bytes_expected = nelm * data_width;

    framefile = fopen(filename, "rb");
    if(framefile == NULL) {
        printf("error: failed to open file: %s\n", filename);
        return 0;
    }

    bytes_read = data_width * fread(array, data_width, nelm, framefile);
    if(bytes_read != bytes_expected) {
        printf("error: reading file: %s, failed, expected: %ld bytes, read: %ld bytes\n",
                filename, bytes_expected, bytes_read);
        return 0;
    }
    return 1;
}

int read_params(char filename[], radar_params_t *params)
{
    printf("Reading parameter data...\n");
    FILE *framefile;
    static const uint8_t data_width = sizeof(radar_params_t);
    size_t bytes_read;
    size_t bytes_expected = data_width;

    framefile = fopen(filename, "rb");
    if(framefile == NULL) {
        printf("error: failed to open file: %s\n", filename);
        return 0;
    }
    bytes_read = data_width * fread(params, data_width, 1, framefile);
    if(bytes_read != bytes_expected) {
        printf("error: reading file: %s, failed, expected: %ld bytes, read: %ld bytes\n",
                filename, bytes_expected, bytes_read);
        return 0;
    }

    return 1;
}

int benchmark1_2_data_gen(
        char *raw_file, 
        char *ldr_file, 
        char rand_data, 
        unsigned int height, 
        unsigned int width
        )
{
    float *raw;
    framefp_t *patches;
    radar_params_t *params;

    /* Allocate parameters buffer */
    printf("Allocating parameters buffer...\n");
    params = (radar_params_t*) malloc(sizeof(radar_params_t));
    if (!params) return 1;

    if (rand_data) gen_params(params, height, width);
    else if(!read_params(ldr_file, params)) return 3;

    /* Allocate raw data buffer */
    unsigned int lines = params->apatch;
    printf("Allocating raw data buffer...\n");
    raw = (float*) malloc(params->rsize*params->asize*2 * sizeof(float)); //Times 2 to accomodate complex values
    if (!raw) return 1;

    /* Writing data in raw buffer */
    if(rand_data) gen_data(raw, params->asize, params->rsize);
    else if(!read_raw(raw_file, raw, params->asize, params->rsize)) return 3;

    /* Allocate data structure */
    printf("Allocating patches buffer...\n");
    patches = (framefp_t *) malloc(params->npatch*sizeof(framefp_t));
    for(int i=0; i<params->npatch; i++)
        if(!framefp_alloc(&patches[i], lines, params->rsize*2)) return 2; 

    printf("Copying raw data to patches...\n");
    for(int i=0; i<params->npatch; i++)
        for(int x = 0; x < (params->rsize<<1); x++)
            for(int y = 0; y < params->apatch; y++){
                int ind = (params->rsize<<1)*(i*params->avalid+y)+x;
                if((ind/(params->rsize<<1))<params->asize) PIXEL(&patches[i],y,x) = raw[ind];
                else PIXEL(&patches[i],y,x) = 0;
            }

    /* Write encryption data to files */
    if(!benchmark1_2_write_files(patches, params)) {
        /* Free buffers if error happen */
        free(raw);
        for(int i=0; i<params->npatch; i++)
            framefp_free(&patches[i]);
        free(patches);
        free(params);
        return 3;
    }

    /* Free buffers */
    free(raw);
    for(int i=0; i<params->npatch; i++)
        framefp_free(&patches[i]);
    free(patches);
    free(params);

    printf("Done.\n");

    return 0; 
}

/* Main */

int main(int argc, char *argv[])
{
    int ret;
    srand (28012015);
    /* Settings */
    unsigned int width = 0, height = 0;
    char raw_file[100], ldr_file[100];
    char rand_data = 0;
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Arguments  
    ///////////////////////////////////////////////////////////////////////////////////////////////
    int resolution = arguments_handler(argc, argv, &rand_data, &height, &width, raw_file, ldr_file);
    if (resolution == ERROR_ARGUMENTS){
        exit(-1);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Data generation
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ret = benchmark1_2_data_gen(raw_file, ldr_file, rand_data, height, width);
    if(ret != 0) return ret;

    return 0;
}

void print_usage(const char * appName)
{
    printf("Usage(1): %s -i raw_file -p param_file [-h]\n", appName);
    printf("Usage(2): %s -r -h Size -w Size [-h]\n\n", appName);
    printf(" -i: input raw data file \n");
    printf(" -p: input parameters (ldr) file \n");
    printf(" -r: use random data (discards -i and -p) \n");
    printf(" -h: height of the input data (if random data)\n");
    printf(" -w: Width of the imput data (if random data) \n");
}

int arguments_handler(int argc, char ** argv, char *rand_data, unsigned int *height, unsigned int *width, char *raw_file, char *ldr_file){
    if (argc == 1){
        printf("-i, -p or -r, -h, -w need to be set\n\n");
        print_usage(argv[0]);
        return ERROR_ARGUMENTS;
    } 
    for(unsigned int args = 1; args < argc; ++args)
    {
        switch (argv[args][1]) {
            case 'i' : args +=1; strcpy(raw_file, argv[args]);break;
            case 'p' : args +=1; strcpy(ldr_file, argv[args]);break;
            case 'h' : args +=1; *height = atoi(argv[args]);break;
            case 'w' : args +=1; *width = atoi(argv[args]);break;
            case 'r' : *rand_data = 1;break;
            default: print_usage(argv[0]); return ERROR_ARGUMENTS;
        }

    }
    if (*rand_data){
        if (*height == 0 || *width == 0){
            printf("\nif random, -h and -w need to be set\n\n");
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


int write_params(char filename[], radar_params_t *array)
{
    FILE *framefile;
    size_t bytes_written;
    static const int data_width = (int)sizeof(radar_params_t);
    size_t bytes_expected = data_width;

    framefile = fopen(filename, "wb");
    if(framefile == NULL) {
        printf("error: failed to open file: %s\n", filename);
        return 0;
    }

    bytes_written = data_width * fwrite(array, data_width, 1, framefile);
    if(bytes_written != bytes_expected) {
        printf("error: writing file: %s, failed, expected: %ld bytes, wrote: %ld bytes\n",
                filename, bytes_expected, bytes_written);
        return 0;
    }

    fclose(framefile);
    printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_written, filename, bytes_expected);

    return 1;
}

