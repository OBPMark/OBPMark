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
    hipSetDevice(device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    strcpy(device_name,prop.name);
    //event create 
    t->start = new hipEvent_t;
    t->stop = new hipEvent_t;
    t->start_memory_copy_device = new hipEvent_t;
    t->stop_memory_copy_device = new hipEvent_t;
    t->start_memory_copy_host = new hipEvent_t;
    t->stop_memory_copy_host= new hipEvent_t;
    
    hipEventCreate(t->start);
    hipEventCreate(t->stop);
    hipEventCreate(t->start_memory_copy_device);
    hipEventCreate(t->stop_memory_copy_device);
    hipEventCreate(t->start_memory_copy_host);
    hipEventCreate(t->stop_memory_copy_host);

}


bool device_memory_init(
	AES_data_t *AES_data,
    unsigned int key_size,
    unsigned int data_length
	)
{	
    hipError_t err = hipSuccess;

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
    err = hipMalloc((void **)&(AES_data->host_key->value), sizeof(uint8_t)*key_size/8);
    if (err != hipSuccess) return false;

    //set data length
    AES_data->host->data_length = (size_t) data_length;

    //All pointers are in GPU memory

    /* memory for key */
    err = hipMalloc((void **)&(AES_data->host->key), sizeof(AES_key_t));
    if (err != hipSuccess) return false;

    /* memory for plaintext */
    err = hipMalloc((void **)&(AES_data->host->plaintext), sizeof(uint8_t)*data_length);
    if (err != hipSuccess) return false;

    /* memory for cyphertext */
    err = hipMalloc((void **)&(AES_data->host->cyphertext), sizeof(uint8_t)*data_length);
    if (err != hipSuccess) return false;

    /* memory for initialization vector */
    err = hipMalloc((void **)&(AES_data->host->iv), sizeof(uint8_t)*data_length);
    if (err != hipSuccess) return false;

    /* allocate constant lookup tables */
    /* memory for sbox (256 uint8) */
    err = hipMalloc((void **)&(AES_data->host->sbox), sizeof(uint8_t)*256);
    if (err != hipSuccess) return false;

    /* memory for rcon (11 uint8) */
    err = hipMalloc((void **)&(AES_data->host->rcon), sizeof(uint8_t)*11);
    if (err != hipSuccess) return false;

    /* memory for roundkey (expanded key Nb*(Nr+1) uint32) */
    err = hipMalloc((void **)&(AES_data->host->expanded_key), sizeof(uint32_t)*AES_data->host_key->Nb*(AES_data->host_key->Nr+1));
    if (err != hipSuccess) return false;

    // memory for device data
    err = hipMalloc((void **)&(AES_data->dev), sizeof(AES_values_t));
    if (err != hipSuccess) return false;

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
    hipEventRecord(*t->start_memory_copy_device);

    /* initialize key value */
    hipMemcpy(AES_data->host_key->value, input_key, sizeof(uint8_t)*AES_data->host_key->size/8, hipMemcpyHostToDevice);

    /* copy key from host to device */
    hipMemcpy(AES_data->host->key, AES_data->host_key, sizeof(AES_key_t), hipMemcpyHostToDevice);

    /* initialize input text */
    hipMemcpy(AES_data->host->plaintext, input_text, sizeof(uint8_t)*AES_data->host->data_length, hipMemcpyHostToDevice);

    /* initialize iv */
    hipMemcpy(AES_data->host->iv, input_iv, sizeof(uint8_t)*16, hipMemcpyHostToDevice);

    /* initialize sbox */
    hipMemcpy(AES_data->host->sbox, input_sbox, sizeof(uint8_t)*256, hipMemcpyHostToDevice);

    /* initialize rcon */
    hipMemcpy(AES_data->host->rcon, input_rcon, sizeof(uint8_t)*11, hipMemcpyHostToDevice);

    /* copy data host to device */
    hipMemcpy(AES_data->dev, AES_data->host, sizeof(AES_values_t), hipMemcpyHostToDevice);

    hipEventRecord(*t->stop_memory_copy_device);
}


void process_benchmark(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{    
    int data_block = AES_data->host->data_length/(4*AES_data->host_key->Nb);
    int n_blocks = data_block/BLOCK_SIZE+(data_block%BLOCK_SIZE==0?0:1);
#ifdef CUDA_FINE
    dim3 threads(BLOCK_SIZE,AES_data->host_key->Nb,4);
#else 
    dim3 threads(BLOCK_SIZE,1,1);
#endif
    int key_threads = AES_data->host_key->Nk;
    hipEventRecord(*t->start);
    hipLaunchKernelGGL(AES_KeyExpansion, dim3(1), dim3(key_threads), 0, 0, AES_data->dev);
    hipDeviceSynchronize();
    hipLaunchKernelGGL(AES_encrypt, dim3(n_blocks), dim3(threads), 0, 0, AES_data->dev);
    hipDeviceSynchronize();
    hipEventRecord(*t->stop);
}

void copy_memory_to_host(
	AES_data_t *AES_data,
    AES_time_t *t,
	uint8_t *output
	)
{
   hipEventRecord(*t->start_memory_copy_host);
   hipMemcpy(output, AES_data->host->cyphertext, sizeof(uint8_t)*AES_data->host->data_length, hipMemcpyDeviceToHost);
   hipEventRecord(*t->stop_memory_copy_host);
}

void get_elapsed_time(
	AES_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{	
    hipEventSynchronize(*t->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    hipEventElapsedTime(&milliseconds_h_d, *t->start_memory_copy_device, *t->stop_memory_copy_device);
    // kernel time
    hipEventElapsedTime(&milliseconds, *t->start, *t->stop);
    //  memory transfer time device-host
    hipEventElapsedTime(&milliseconds_d_h, *t->start_memory_copy_host, *t->stop_memory_copy_host);

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
    hipError_t err = hipSuccess;
	/* Clean time */
	err = hipFree(AES_data->dev);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device data (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}

	err = hipFree(AES_data->host->key);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device key (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->plaintext);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device input text (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->expanded_key);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device expanded key (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->cyphertext);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device encrypted text (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->iv);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device initialization vector (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->sbox);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device sbox (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host->rcon);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device rcon (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	err = hipFree(AES_data->host_key->value);
	if(err != hipSuccess) {
	    fprintf(stderr, "Failed to free device key value (errer code %s)!\n", hipGetErrorString(err)); 
	    return;
	}
	free(AES_data->host_key);
	free(AES_data->host);
    free(AES_data);
    free(t);

}
