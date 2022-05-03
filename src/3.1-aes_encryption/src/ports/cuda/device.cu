/**
 * \file device.c
 * \brief Benchmark #3.1 CPU version (sequential) device initialization. 
 * \author Marc Sole (BSC)
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
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    strcpy(device_name,prop.name);
    //event create 
    t->start = new cudaEvent_t;
    t->stop = new cudaEvent_t;
    t->start_memory_copy_device = new cudaEvent_t;
    t->stop_memory_copy_device = new cudaEvent_t;
    t->start_memory_copy_host = new cudaEvent_t;
    t->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(t->start);
    cudaEventCreate(t->stop);
    cudaEventCreate(t->start_memory_copy_device);
    cudaEventCreate(t->stop_memory_copy_device);
    cudaEventCreate(t->start_memory_copy_host);
    cudaEventCreate(t->stop_memory_copy_host);

}


bool device_memory_init(
	AES_data_t *AES_data,
    unsigned int key_size,
    unsigned int data_length
	)
{	
    cudaError_t err = cudaSuccess;

    //malloc host addresses
    AES_data->host = (AES_values_t*) malloc(sizeof(AES_values_t));
    AES_data->host_key = (AES_key_t*) malloc(sizeof(AES_key_t));

    //fill host_key value
    AES_data->host_key->Nb = 4;
    AES_data->host_key->size = (AES_keysize_t) key_size;
    switch(key_size) {
        case AES_KEY128: AES_data->host_key->Nk = 4; AES_data->host_key->Nr = 10; break;
        case AES_KEY192: AES_data->host_key->Nk = 6; AES_data->host_key->Nr = 12; break;
        case AES_KEY256: AES_data->host_key->Nk = 8; AES_data->host_key->Nr = 14; break;
    }

    //host_key points to gpu mem
    err = cudaMalloc((void **)&(AES_data->host_key->value), sizeof(uint8_t)*key_size/8);
    if (err != cudaSuccess) return false;

    //set data length
    AES_data->host->data_length = (size_t) data_length;

    //All pointers are in GPU memory

    /* memory for key */
    err = cudaMalloc((void **)&(AES_data->host->key), sizeof(AES_key_t));
    if (err != cudaSuccess) return false;

    /* memory for plaintext */
    err = cudaMalloc((void **)&(AES_data->host->plaintext), sizeof(uint8_t)*data_length);
    if (err != cudaSuccess) return false;

    /* memory for cyphertext */
    err = cudaMalloc((void **)&(AES_data->host->cyphertext), sizeof(uint8_t)*data_length);
    if (err != cudaSuccess) return false;

    /* memory for initialization vector */
    err = cudaMalloc((void **)&(AES_data->host->iv), sizeof(uint8_t)*data_length);
    if (err != cudaSuccess) return false;

    /* allocate constant lookup tables */
    /* memory for sbox (256 uint8) */
    err = cudaMalloc((void **)&(AES_data->host->sbox), sizeof(uint8_t)*256);
    if (err != cudaSuccess) return false;

    /* memory for rcon (11 uint8) */
    err = cudaMalloc((void **)&(AES_data->host->rcon), sizeof(uint8_t)*11);
    if (err != cudaSuccess) return false;

    /* memory for roundkey (expanded key Nb*(Nr+1) uint32) */
    err = cudaMalloc((void **)&(AES_data->host->expanded_key), sizeof(uint32_t)*AES_data->host_key->Nb*(AES_data->host_key->Nr+1));
    if (err != cudaSuccess) return false;

    // memory for device data
    err = cudaMalloc((void **)&(AES_data->dev), sizeof(AES_values_t));
    if (err != cudaSuccess) return false;

    return true;
}

void copy_memory_to_device(
	AES_data_t *AES_data,
	AES_time_t *t,
	uint8_t *input_key,
	uint8_t *input_text,
	uint8_t *input_iv,
	uint8_t *input_sbox,
	uint8_t *input_rcon
	)
{
    cudaEventRecord(*t->start_memory_copy_device);

    /* initialize key value */
    cudaMemcpy(AES_data->host_key->value, input_key, sizeof(uint8_t)*AES_data->host_key->size/8, cudaMemcpyHostToDevice);

    /* copy key from host to device */
    cudaMemcpy(AES_data->host->key, AES_data->host_key, sizeof(AES_key_t), cudaMemcpyHostToDevice);

    /* initialize input text */
    cudaMemcpy(AES_data->host->plaintext, input_text, sizeof(uint8_t)*AES_data->host->data_length, cudaMemcpyHostToDevice);

    /* initialize iv */
    cudaMemcpy(AES_data->host->iv, input_iv, sizeof(uint8_t)*16, cudaMemcpyHostToDevice);

    /* initialize sbox */
    cudaMemcpy(AES_data->host->sbox, input_sbox, sizeof(uint8_t)*256, cudaMemcpyHostToDevice);

    /* initialize rcon */
    cudaMemcpy(AES_data->host->rcon, input_rcon, sizeof(uint8_t)*11, cudaMemcpyHostToDevice);

    /* copy data host to device */
    cudaMemcpy(AES_data->dev, AES_data->host, sizeof(AES_values_t), cudaMemcpyHostToDevice);

    cudaEventRecord(*t->stop_memory_copy_device);
}


void process_benchmark(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{    
    int n_threads = AES_data->host->data_length/(4*AES_data->host_key->Nb);
    int n_blocks = n_threads/512+(n_threads%512==0?0:1);
    int key_threads = AES_data->host_key->Nk;
    cudaEventRecord(*t->start);
    AES_KeyExpansion<<<1,key_threads>>>(AES_data->dev);
    cudaDeviceSynchronize();
    AES_encrypt<<<n_blocks, 512>>>(AES_data->dev);
    cudaDeviceSynchronize();
    cudaEventRecord(*t->stop);
}

void copy_memory_to_host(
	AES_data_t *AES_data,
    AES_time_t *t,
	uint8_t *output
	)
{
   cudaEventRecord(*t->start_memory_copy_host);
   cudaMemcpy(output, AES_data->host->cyphertext, sizeof(uint8_t)*AES_data->host->data_length, cudaMemcpyDeviceToHost);
   cudaEventRecord(*t->stop_memory_copy_host);
}

void get_elapsed_time(
	AES_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{	
    cudaEventSynchronize(*t->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *t->start_memory_copy_device, *t->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *t->start, *t->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *t->start_memory_copy_host, *t->stop_memory_copy_host);

	if (csv_format)
	{
		printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d, milliseconds, milliseconds_d_h);
	}
	else if (database_format)
	{
		printf("%.10f;%.10f;%.10f;%ld;\n", milliseconds_h_d, milliseconds, milliseconds_d_h, timestamp);
	}
	else if(verbose_print)
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", milliseconds_h_d);
		printf("Elapsed time kernel: %.10f milliseconds\n", milliseconds );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", milliseconds_d_h);
	}
}


void clean(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{
    cudaError_t err = cudaSuccess;
	/* Clean time */
	err = cudaFree(AES_data->dev);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device data (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}

	err = cudaFree(AES_data->host->key);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device key (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->plaintext);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device input text (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->expanded_key);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device expanded key (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->cyphertext);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device encrypted text (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->iv);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device initialization vector (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->sbox);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device sbox (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host->rcon);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device rcon (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	err = cudaFree(AES_data->host_key->value);
	if(err != cudaSuccess) {
	    fprintf(stderr, "Failed to free device key value (errer code %s)!\n", cudaGetErrorString(err)); 
	    return;
	}
	free(AES_data->host_key);
	free(AES_data->host);
    free(AES_data);
    free(t);

}
