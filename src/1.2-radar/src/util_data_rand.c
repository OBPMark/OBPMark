/**
 * \file util_data_rand.c 
 * \brief Benchmark #1.2 random data generation.
 * \author Marc Sole Bonet (BSC)
 */

#include "util_data_rand.h"

void benchmark_gen_rand_params(
        radar_params_t *params,
        unsigned int height,
        unsigned int width
)
{
    srand (28012015);
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
    unsigned int i, I = height * width * 2; //times two for complex
    for(i=0; i<I; i++)
        raw[i] = i%50000; //-15.5+31*((float)rand()/(float)(RAND_MAX));
}

void benchmark_gen_rand_data(
        framefp_t *data,
        radar_params_t *params,
        unsigned int height, 
        unsigned int width
        )
{
    float *raw;

    /* Allocate raw data buffer */
    raw = (float*) malloc(width*height*2 * sizeof(float)); //Times 2 to accomodate complex values
    /* Generate random data */
    gen_data(raw, height, width);

    for(int i=0; i<params->npatch; i++){
        for(int x = 0; x < width*2; x++)
            for(int y = 0; y < params->apatch; y++){
                int ind = width*2*(i*params->avalid+y)+x;
                if((ind/(width*2))<params->asize) PIXEL(&data[i],x,y) = raw[ind];
                else PIXEL(&data[i],x,y) = 0;
            }
    }

    free(raw);
}
