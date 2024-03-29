/**
 * \file device.c
 * \brief Benchmark #1.2 GPU version (cuda) device initialization. 
 * \author Marc Sole Bonet (BSC)
 */
#include "device.h"
#include "processing.h"

uint32_t next_power_of_two(uint32_t n)
{
    uint32_t v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void init(
	radar_data_t *radar_data,
	radar_time_t *t,
	char *device_name
	)
{
    init(radar_data,t, 0,0, device_name);
}


void init(
	radar_data_t *radar_data,
	radar_time_t *t,
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
	radar_data_t *radar_data,
	radar_params_t *params,
    unsigned int out_height,
    unsigned int out_width
	)
{	
    unsigned int patch_width = params->rsize<<1;
    unsigned int patch_extended_width = next_power_of_two(params->rsize)<<1;
    unsigned int patch_height = params->apatch;

    radar_data->out_height = out_height;
    radar_data->out_width = out_width;
    radar_data->host_params = params;

#ifdef FFT_LIB
    /* FFT plans */
    cufftPlan1d(&radar_data->rrf_plan, next_power_of_two(params->rsize), CUFFT_C2C, 1);
    cufftPlan1d(&radar_data->arf_plan, params->apatch, CUFFT_C2C, 1);
    cufftPlan1d(&radar_data->range_plan, next_power_of_two(params->rsize), CUFFT_C2C, params->apatch * params->npatch);
    cufftPlan1d(&radar_data->azimuth_plan, params->apatch, CUFFT_C2C, params->rvalid * params->npatch);
#endif

    cudaError_t err = cudaSuccess;

	/* radar_data_t memory allocation */
	//RANGE & AZIMUTH DATA
	err = cudaMalloc((void **)&(radar_data->range_data),
	        sizeof(float) * params->npatch * patch_height * patch_extended_width);
    if (err != cudaSuccess) return false;
    err = cudaMemset(radar_data->range_data, 0,
	        sizeof(float) * params->npatch * patch_height * patch_extended_width);
    if (err != cudaSuccess) return false;

	err = cudaMalloc((void **)&(radar_data->azimuth_data),
	        sizeof(float) * params->npatch * patch_height * patch_width);
    if (err != cudaSuccess) return false;
    err = cudaMemset(radar_data->azimuth_data, 0,
	        sizeof(float) * params->npatch * patch_height * patch_width);
    if (err != cudaSuccess) return false;

  	//MULTILOOK DATA
	err = cudaMalloc((void **)&(radar_data->ml_data),
	        sizeof(float) * params->npatch * out_height * out_width);
    if (err != cudaSuccess) return false;

  	//OUTPUT DATA
	err = cudaMalloc((void **)&(radar_data->output_image),
	        sizeof(uint8_t) * params->npatch * out_height * out_width);
    if (err != cudaSuccess) return false;

  	//PARAMS
	err = cudaMalloc((void **)&(radar_data->params), sizeof(radar_params_t));
    if (err != cudaSuccess) return false;
    
    //RANGE REF. FUNCTION
	err = cudaMalloc((void **)&(radar_data->rrf), sizeof(float) * patch_extended_width);
    if (err != cudaSuccess) return false;
    err = cudaMemset(radar_data->rrf, 0, sizeof(float) * patch_extended_width);
    if (err != cudaSuccess) return false;

	//AZIMUTH REF. FUNCTION
	err = cudaMalloc((void **)&(radar_data->arf), sizeof(float) * (patch_height<<1));
    if (err != cudaSuccess) return false;
    err = cudaMemset(radar_data->arf, 0, sizeof(float) * (patch_height<<1));
    if (err != cudaSuccess) return false;

    //RCMC TABLE
	err = cudaMalloc((void **)&(radar_data->offsets), sizeof(uint32_t) * params->rvalid * patch_height);
    if (err != cudaSuccess) return false;


    return true;
}

void copy_memory_to_device(
	radar_data_t *radar_data,
	radar_time_t *t,
	framefp_t *input_data,
	radar_params_t *input_params
	)
{
    cudaEventRecord(*t->start_memory_copy_device);

    /* Copy params */
    cudaMemcpy(radar_data->params, input_params, sizeof(radar_params_t), cudaMemcpyHostToDevice);
    uint32_t width = input_params->rsize<<1;
    uint32_t height = input_params->apatch;
    uint32_t line_width = next_power_of_two(width);
    uint32_t patch_size = line_width * height;
    for (uint32_t i = 0; i < input_params->npatch; i++ )
        for(uint32_t j = 0; j < height; j++){
            uint32_t offs = i * patch_size + j * line_width;
            cudaMemcpy(&radar_data->range_data[offs], &input_data[i].f[j * width], width * sizeof(float), cudaMemcpyHostToDevice);
        }

    cudaEventRecord(*t->stop_memory_copy_device);
}

#ifndef FFT_LIB
const int FFT_FORWARD = 1;
const int FFT_INVERSE = -1;
void launch_fft(float *data, unsigned int width, unsigned int rows, unsigned int npatch, int direction)
{
    unsigned int group = (unsigned int) log2(width);
    unsigned int nthreads = width>>1;
    dim3 gridSize((width-1)/BLOCK_SIZE+1,rows,npatch); 
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    bin_reverse<<<gridSize, blockSize>>>(data, width, group);

    float wtemp, wpr, wpi, theta;
    int loop = 1;

    if (nthreads % BLOCK_SIZE != 0){
            // inferior part
            blockSize.x = nthreads;
            gridSize.x  = 1;
    }
    else{
            // top part
            blockSize.x = BLOCK_SIZE;
            gridSize.x  = (unsigned int)(nthreads/BLOCK_SIZE);
    }
    while(loop < width){
        // calculate values
        theta = -M_PI/(loop*direction); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        //kernel launch
        fft_kernel<<<gridSize, blockSize>>>(data, loop, wpr, wpi, nthreads, width);
        // update loop values
        loop = loop * 2;
    }
}
#endif

void process_benchmark(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{    
    cudaDeviceSynchronize();
    cudaEventRecord(*t->start);

    radar_params_t *params = radar_data->host_params;

    /* SAR RANGE REFERENCE */
    int n_blocks = (params->rsize-1)/BLOCK_SIZE+1;
    // compute reference function
    uint32_t nit = floor(params->tau * params->fs);
    SAR_range_ref<<<n_blocks,BLOCK_SIZE>>>(radar_data->rrf, radar_data->params, nit);
    // perform fft
#ifdef FFT_LIB
    cufftExecC2C(radar_data->rrf_plan, (cufftComplex*) radar_data->rrf, (cufftComplex*) radar_data->rrf, CUFFT_FORWARD);
#else
    launch_fft(radar_data->rrf, next_power_of_two(params->rsize), 1, 1, FFT_FORWARD);
#endif


    /* DOPPLER CENTROID */
    dim3 blockSize(TILE_SIZE,TILE_SIZE,1);
    float const_k = params->PRF/(2*pi*params->rsize);
    SAR_DCE<<<params->rsize,BLOCK_SIZE,sizeof(float)*2*params->apatch>>>(radar_data->range_data, radar_data->params, const_k);

    /* RCMC table */
    dim3 gridSize(params->apatch/TILE_SIZE,(params->rvalid-1)/TILE_SIZE+1,1);
    SAR_rcmc_table<<<gridSize, blockSize>>>(radar_data->params, radar_data->offsets);

    /* SAR AZIMUTH REFERENCE */
    // compute azimuth values
    n_blocks = (params->apatch)/BLOCK_SIZE;
    // Compute azimuth reference
    SAR_azimuth_ref<<<n_blocks, BLOCK_SIZE>>>(radar_data->arf, radar_data->params);
    // perform fft
#ifdef FFT_LIB
    cufftExecC2C(radar_data->arf_plan, (cufftComplex*) radar_data->arf, (cufftComplex*) radar_data->arf, CUFFT_FORWARD);
#else
    launch_fft(radar_data->arf, params->apatch, 1, 1, FFT_FORWARD);
#endif

    /* Begin patch processing */
    //SAR Range Compress
#ifdef FFT_LIB
    cufftExecC2C(radar_data->range_plan, (cufftComplex*) radar_data->range_data, (cufftComplex*) radar_data->range_data, CUFFT_FORWARD);
#else
    launch_fft(radar_data->range_data, next_power_of_two(params->rsize), params->apatch, params->npatch, FFT_FORWARD);
#endif
    gridSize = {next_power_of_two(params->rsize)/TILE_SIZE, params->apatch/TILE_SIZE, params->npatch};
    SAR_ref_product<<<gridSize,blockSize>>>(radar_data->range_data, radar_data->rrf, next_power_of_two(params->rsize), params->apatch);
#ifdef FFT_LIB
    cufftExecC2C(radar_data->range_plan, (cufftComplex*) radar_data->range_data, (cufftComplex*) radar_data->range_data, CUFFT_INVERSE);
#else
    launch_fft(radar_data->range_data, next_power_of_two(params->rsize), params->apatch, params->npatch, FFT_INVERSE);
#endif
    //after IFFT data needs to be idvided by next_power_of_two(rsize), we do that when transposing
    SAR_transpose<<<gridSize, blockSize>>>(radar_data->range_data, radar_data->azimuth_data, next_power_of_two(params->rsize), params->apatch, params->apatch, params->rvalid);
#ifdef FFT_LIB
    cufftExecC2C(radar_data->azimuth_plan, (cufftComplex*) radar_data->azimuth_data, (cufftComplex*) radar_data->azimuth_data, CUFFT_FORWARD);
#else
    launch_fft(radar_data->azimuth_data, params->apatch, params->rvalid, params->npatch, FFT_FORWARD);
#endif

    /* RCMC */
    blockSize = {BLOCK_SIZE,1,1};
    gridSize= {(params->apatch)/BLOCK_SIZE,params->npatch,1};
    SAR_rcmc<<<gridSize,blockSize>>>(radar_data->azimuth_data , radar_data->offsets, params->apatch, params->rvalid);

    /* RCMC */
    /* Azimuth Compress */
    blockSize = {TILE_SIZE,TILE_SIZE,1};
    gridSize= {params->apatch/TILE_SIZE,(params->rvalid-1)/TILE_SIZE+1,params->npatch};
    SAR_ref_product<<<gridSize, blockSize>>>(radar_data->azimuth_data, radar_data->arf, params->apatch, params->rvalid);
#ifdef FFT_LIB
    cufftExecC2C(radar_data->azimuth_plan, (cufftComplex*) radar_data->azimuth_data, (cufftComplex*) radar_data->azimuth_data, CUFFT_INVERSE);
#else
    launch_fft(radar_data->azimuth_data, params->apatch, params->rvalid, params->npatch, FFT_INVERSE);
#endif
    //after IFFT data needs to be idvided by next_power_of_two(rsize), we do that when transposing
    SAR_transpose<<<gridSize, blockSize>>>(radar_data->azimuth_data, radar_data->range_data, params->apatch, next_power_of_two(params->rsize), params->rvalid, params->apatch);

    gridSize= {(radar_data->out_width-1)/TILE_SIZE+1,(radar_data->out_height-1)/TILE_SIZE+1,1};
    SAR_multilook<<<gridSize,blockSize>>>(radar_data->range_data, radar_data->ml_data, radar_data->params, radar_data->out_width, radar_data->out_height);
    quantize<<<gridSize,blockSize>>>(radar_data->ml_data, radar_data->output_image, radar_data->out_width, radar_data->out_height);

    cudaEventRecord(*t->stop);
}

void copy_memory_to_host(
	radar_data_t *radar_data,
	radar_time_t *t,
	frame8_t *output_radar
	)
{
    cudaEventRecord(*t->start_memory_copy_host);
    uint32_t  width = output_radar->w;
    uint32_t  height = output_radar->h;
    cudaMemcpy(output_radar->f, radar_data->output_image, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);
    cudaEventRecord(*t->stop_memory_copy_host);
}


void get_elapsed_time(
	radar_data_t *radar_data, 
	radar_time_t *t, 
    print_info_data_t *benchmark_info,
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

    print_execution_info(benchmark_info, true, timestamp,milliseconds_h_d,milliseconds,milliseconds_d_h);
}


void clean(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{

	/* Clean time */
	free(t);

    cudaError_t err = cudaSuccess;

	/* Clean radar data */
	err = cudaFree(radar_data->range_data);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->azimuth_data);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->ml_data);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->output_image);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->params);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->rrf);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->arf);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

	err = cudaFree(radar_data->offsets);
	if(err != cudaSuccess) { fprintf(stderr, "Failed to free device data (error code %s)!\n", cudaGetErrorString(err)); return; }

    free(radar_data);
}
