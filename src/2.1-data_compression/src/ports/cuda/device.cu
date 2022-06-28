/**
 * \file device.c
 * \brief Benchmark #121 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
 #include "device.h"
 #include "processing.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d code id: %d\n", cudaGetErrorString(code), file, line, code);
      if (abort) exit(code);
   }
}
 
 cudaStream_t cuda_streams[NUMBER_STREAMS][MAXSIZE_NBITS];

 void init(
     compression_data_t *compression_data,
     compression_time_t *t,
     char *device_name
     )
 {
     init(compression_data,t, 0,0, device_name);
 }
 
 
 
 void init(
     compression_data_t *compression_data,
     compression_time_t *t,
     int platform,
     int device,
     char *device_name
    ) 
 {
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    t->start_memory_copy_device = new cudaEvent_t;
    t->stop_memory_copy_device = new cudaEvent_t;
    t->start_memory_copy_host = new cudaEvent_t;
    t->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(t->start_memory_copy_device);
    cudaEventCreate(t->stop_memory_copy_device);
    cudaEventCreate(t->start_memory_copy_host);
    cudaEventCreate(t->stop_memory_copy_host);
}


bool device_memory_init(
	compression_data_t *compression_data
	)
{	
	// Allocate the device image imput
    cudaError_t err = cudaSuccess;
    // data input
 
    err = cudaMalloc((void **)&compression_data->input_data, sizeof( unsigned int  ) * compression_data->TotalSamplesStep);
    if (err != cudaSuccess)
    {
        return false;
    }
    // data post_processed
    err = cudaMalloc((void **)&compression_data->input_data_post_process, sizeof( unsigned int  ) * compression_data->TotalSamples * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }
    // data out

    err = cudaMalloc((void **)&compression_data->output_data, sizeof( unsigned int  ) * compression_data->TotalSamplesStep);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->missing_value, sizeof(int) * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->missing_value_inverse, sizeof(int) * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->zero_block_list, sizeof(int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->zero_block_list_inverse, sizeof(int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }
    // Aee variables 
    err = cudaMalloc((void **)&compression_data->size_block, sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->compresion_identifier, sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->compresion_identifier_internal, sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->halved_samples, sizeof(unsigned int) * compression_data->j_blocksize/2 *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->data_in_blocks_best, sizeof(unsigned int )  * compression_data->r_samplesInterval * compression_data->j_blocksize *  NUMBER_STREAMS );
    if (err != cudaSuccess)
    {
        return false;
    }

    /*err = cudaMalloc((void **)&compression_data->data_in_blocks_best_post_process, sizeof(unsigned int )  * compression_data->r_samplesInterval * compression_data->j_blocksize *  NUMBER_STREAMS );
    if (err != cudaSuccess)
    {
        return false;
    }*/

    err = cudaMalloc((void **)&compression_data->size_block_best, sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->bit_block_best, sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->compresion_identifier_best, sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->compresion_identifier_internal_best, sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&compression_data->data_in_blocks, sizeof(unsigned int ) * compression_data->r_samplesInterval * compression_data->j_blocksize *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    // INIT STREAMS
    for(unsigned int x = 0; x < NUMBER_STREAMS; ++x){
        for(unsigned int y = 0; y < 2 + compression_data->n_bits; ++ y){
            cudaStreamCreate(&cuda_streams[x][y]);
        }
    }

    // int CPU part
    compression_data->data_in_blocks_best_cpu = ( unsigned int *)malloc(sizeof( unsigned int ) * compression_data->TotalSamplesStep);
    compression_data->size_block_best_cpu = ( unsigned int *)malloc(sizeof(unsigned int) * compression_data->r_samplesInterval * compression_data->steps);
    compression_data->compresion_identifier_best_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * compression_data->r_samplesInterval * compression_data->steps);
    compression_data->compresion_identifier_best_internal_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * compression_data->r_samplesInterval * compression_data->steps);

	return true;
}



void copy_memory_to_device(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
    cudaEventRecord(*t->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(compression_data->input_data, compression_data->InputDataBlock, sizeof( unsigned int  ) * compression_data->TotalSamplesStep, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input_image from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
	cudaEventRecord(*t->stop_memory_copy_device);
}


void copy_data_to_cpu_asynchronous(
    compression_data_t *compression_data,
	compression_time_t *t, 
    int step)
{
    cudaEventRecord(*t->start_memory_copy_host);
    cudaMemcpyAsync(compression_data->data_in_blocks_best_cpu + (compression_data->j_blocksize * compression_data->r_samplesInterval * step), compression_data->data_in_blocks_best + ((step % NUMBER_STREAMS) * compression_data->j_blocksize * compression_data->r_samplesInterval), sizeof( unsigned  int ) * compression_data->r_samplesInterval * compression_data->j_blocksize, cudaMemcpyDeviceToHost, cuda_streams[step % NUMBER_STREAMS][0]);
    cudaMemcpyAsync(compression_data->size_block_best_cpu + (compression_data->r_samplesInterval * step), compression_data->size_block_best + ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval), sizeof( unsigned int ) * compression_data->r_samplesInterval , cudaMemcpyDeviceToHost,cuda_streams[step % NUMBER_STREAMS][1]);
    cudaMemcpyAsync(compression_data->compresion_identifier_best_cpu + (compression_data->r_samplesInterval * step), compression_data->compresion_identifier_best + ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval), sizeof( unsigned char ) * compression_data->r_samplesInterval , cudaMemcpyDeviceToHost,cuda_streams[step % NUMBER_STREAMS][2]);
    cudaMemcpyAsync(compression_data->compresion_identifier_best_internal_cpu + (compression_data->r_samplesInterval * step), compression_data->compresion_identifier_internal_best + ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval), sizeof( unsigned char ) * compression_data->r_samplesInterval , cudaMemcpyDeviceToHost,cuda_streams[step % NUMBER_STREAMS][3]);
    
    cudaEventRecord(*t->stop_memory_copy_host);

}

void process_benchmark(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
   
    T_START(t->t_test);

    // Repeating the operations n times
    for(int step = 0; step < compression_data->steps; ++step)
    {
        // check if preprocessing is required
        if(compression_data->preprocessor_active)
        {
            // Preprocesor active
            dim3 dimBlock_prepro(BLOCK_SIZE*BLOCK_SIZE);
            dim3 dimGrid_prepro(ceil(float(compression_data->r_samplesInterval)/dimBlock_prepro.x));
            process_input_preprocessor<<<dimGrid_prepro,dimBlock_prepro,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->input_data + (compression_data->TotalSamples * step),
            compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)) ,
            compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->zero_block_list_inverse + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->j_blocksize,
            compression_data->r_samplesInterval,
            compression_data->n_bits);
        }
        else
        {
            // Not preprocessor
            dim3 dimBlock_no_prepro(BLOCK_SIZE*BLOCK_SIZE);
            dim3 dimGrid_no_prepro(ceil(float(compression_data->r_samplesInterval)/dimBlock_no_prepro.x));
            process_input_no_preprocessor<<<dimGrid_no_prepro,dimBlock_no_prepro,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->input_data + (compression_data->TotalSamples * step),
                compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)) ,
                compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
                compression_data->zero_block_list_inverse + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
                compression_data->j_blocksize,
                compression_data->r_samplesInterval);
        }
        gpuErrchk( cudaPeekAtLastError() );
        // copy data and detect the zero blockls
        dim3 dimBlock_zero(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid_zero(ceil(float((compression_data->r_samplesInterval)/2)/dimBlock_zero.x));
        // identify the total remaining zero blocks
        zero_block_list_completition<<<dimGrid_zero,dimBlock_zero,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->zero_block_list+ (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->zero_block_list_inverse+ (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->missing_value,
             compression_data->missing_value_inverse,
            (step % NUMBER_STREAMS),
            compression_data->j_blocksize,
            compression_data->r_samplesInterval);
        // zero block finish 
        gpuErrchk( cudaPeekAtLastError() );
        // sync stream
        cudaStreamSynchronize (cuda_streams[step % NUMBER_STREAMS][0]);
        dim3 dimBlock;
        dim3 dimGrid;
        
        // start  adaptative entropy encoder
        dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
        dimGrid = dim3(ceil(float(compression_data->r_samplesInterval)/dimBlock.x), ceil(float(compression_data->j_blocksize)/dimBlock.y));
        adaptative_entropy_encoder_no_compresion<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)), 
            compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) ) ,
            compression_data->data_in_blocks + (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->size_block + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier_internal + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            0,
            compression_data->j_blocksize,
            compression_data->r_samplesInterval,
            compression_data->n_bits);
        gpuErrchk( cudaPeekAtLastError() );
        
        dimBlock = dim3(BLOCK_SIZE*BLOCK_SIZE);
        dimGrid = dim3(ceil(float(compression_data->r_samplesInterval)/dimBlock.x));
        adaptative_entropy_encoder_zero_block<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)), 
            compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) ) ,
            compression_data->zero_block_list_inverse + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) ) , 
            compression_data->data_in_blocks + (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->size_block + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier_internal + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            0,
            compression_data->j_blocksize,
            compression_data->r_samplesInterval,
            compression_data->n_bits);
        gpuErrchk( cudaPeekAtLastError() );
        // launch second extension
       adaptative_entropy_encoder_second_extension<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][1]>>>(compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)), 
            compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) ) , 
            compression_data->data_in_blocks + (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->size_block + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->halved_samples + ((compression_data->j_blocksize)/2 * (step % NUMBER_STREAMS)),
            compression_data->compresion_identifier + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier_internal + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            1,
            compression_data->j_blocksize,
            compression_data->r_samplesInterval,
            compression_data->n_bits);
        gpuErrchk( cudaPeekAtLastError() );

        // launch sample spiting
        for (unsigned int bit = 0; bit < compression_data->n_bits; ++ bit)
        {
            adaptative_entropy_encoder_sample_spliting<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][bit + 2]>>>(compression_data->input_data_post_process + (compression_data->TotalSamples * (step % NUMBER_STREAMS)), 
                compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) ) , 
                compression_data->data_in_blocks + (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
                compression_data->size_block + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
                compression_data->compresion_identifier + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
                compression_data->compresion_identifier_internal + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
                bit + 2,
                compression_data->j_blocksize,
                compression_data->r_samplesInterval,
                compression_data->n_bits);
            //gpuErrchk( cudaPeekAtLastError() );
        }
        
        // sync screams of the same "primary" stream
        gpuErrchk( cudaPeekAtLastError() );
        for(unsigned int y = 0; y < 2 + compression_data->n_bits; ++ y)
        {
            cudaStreamSynchronize(cuda_streams[step % NUMBER_STREAMS][y]);
        }
       
        gpuErrchk( cudaPeekAtLastError() );
        // block selector 
        adaptative_entropy_encoder_block_selector<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)) , 
            compression_data->bit_block_best + (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->size_block + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->compresion_identifier_internal + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),
            compression_data->size_block_best + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->compresion_identifier_best + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->compresion_identifier_internal_best + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->j_blocksize,
            compression_data->r_samplesInterval,
            compression_data->n_bits);
        
        gpuErrchk( cudaPeekAtLastError() );
        dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
        dimGrid = dim3(ceil(float(compression_data->r_samplesInterval)/dimBlock.x),ceil(float(compression_data->j_blocksize)/dimBlock.y));
        adaptative_entropy_encoder_block_selector_data_copy<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(compression_data->zero_block_list + (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)) ,
            compression_data->data_in_blocks + (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)),   
            compression_data->bit_block_best + (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->data_in_blocks_best + (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS)),
            compression_data->j_blocksize,
            compression_data->r_samplesInterval);
        gpuErrchk( cudaPeekAtLastError() );
        // copy back the data
        copy_data_to_cpu_asynchronous(compression_data,t, step);
        
    }
    // sync GPU
    cudaDeviceSynchronize();
    
    // copy the data back and write to final data
    unsigned int compression_technique_identifier_size = 1;


    // define the size of the compression technique identifier base of n_bits size
    if (compression_data->n_bits < 3){
        compression_technique_identifier_size = 1;
    }
    else if (compression_data->n_bits < 5)
    {
        compression_technique_identifier_size = 2;
    }
    else if (compression_data->n_bits <= 8)
    {
        compression_technique_identifier_size = 3;
    }
    else if (compression_data->n_bits <= 16)
    {
        compression_technique_identifier_size = 4;
    }
    else 
    {
        compression_technique_identifier_size = 5;
    }

    // go for all of the steps
    for(int step = 0; step < compression_data->steps; ++step)
    {    
        unsigned int final_compression_technique_identifier_size = compression_technique_identifier_size;
        // go per each block
        for(unsigned int block = 0; block < compression_data->r_samplesInterval; ++block)
        {  
           
            if( compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)] == ZERO_BLOCK_ID ||  compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)] == SECOND_EXTENSION_ID)
            {
                final_compression_technique_identifier_size += 1;
            }
            // header
            const unsigned char CompressionTechniqueIdentifier = compression_data->compresion_identifier_best_cpu[block + (step * compression_data->r_samplesInterval)];
            // print CompressionTechniqueIdentifier
            writeWordChar(compression_data->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);
            // block compression
            const unsigned char best_compression_technique_identifier = compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)];
            unsigned int *data_pointer = compression_data->data_in_blocks_best_cpu + (block * compression_data->j_blocksize + (step * compression_data->r_samplesInterval * compression_data->j_blocksize));
            const unsigned int size = compression_data->size_block_best_cpu[block + (step *  compression_data->r_samplesInterval)];


            //printf("%u %d %d %d %d\n", best_compression_technique_identifier , block, step, size,  block * compression_data->j_blocksize + (step * compression_data->r_samplesInterval * compression_data->j_blocksize));

            if (best_compression_technique_identifier == ZERO_BLOCK_ID)
            {
                ZeroBlockWriter(compression_data->OutputDataBlock, size);
            }
            else if(best_compression_technique_identifier == NO_COMPRESSION_ID)
            {
                NoCompressionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, compression_data->n_bits,data_pointer);
            }
            else if(best_compression_technique_identifier == FUNDAMENTAL_SEQUENCE_ID)
            {
                FundamentalSequenceWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, data_pointer);
            }
            else if(best_compression_technique_identifier == SECOND_EXTENSION_ID)
            {
                SecondExtensionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize/2,data_pointer);
            }
            else if(best_compression_technique_identifier >= SAMPLE_SPLITTING_ID)
            {
                SampleSplittingWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, best_compression_technique_identifier - SAMPLE_SPLITTING_ID, data_pointer);
            }
            else
            {
                printf("Error: Unknown compression technique identifier\n");
            }
        }
        
       
      
    }

    T_STOP(t->t_test);


}



void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
//EMPTY In dis case not use because the copy happens asynchronous
}


void get_elapsed_time(
	compression_data_t *compression_data, 
	compression_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{
	    //cudaEventSynchronize(*compression_data->stop_memory_copy_host);
        float milliseconds_h_d = 0, milliseconds_d_h = 0;
        // memory transfer time host-device
        cudaEventElapsedTime(&milliseconds_h_d, *t->start_memory_copy_device, *t->stop_memory_copy_device);
        // kernel time 1
        long unsigned int application_miliseconds = (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
        //  memory transfer time device-host
        cudaEventElapsedTime(&milliseconds_d_h, *t->start_memory_copy_host, *t->stop_memory_copy_host);
        
        if (csv_format)
        {
            printf("%.10f;%lu;%.10f;\n", milliseconds_h_d,application_miliseconds,milliseconds_d_h);
        }
        else if (database_format)
        {
            printf("%.10f;%lu;%.10f;%ld;\n", milliseconds_h_d,application_miliseconds,milliseconds_d_h, timestamp);
        }
        else if(verbose_print)
        {
            printf("Elapsed time Host->Device: %.10f milliseconds\n", (float) milliseconds_h_d);
            printf("Elapsed time kernel: %lu milliseconds\n", application_miliseconds );
            printf("Elapsed time Device->Host: %.10f milliseconds\n", (float) milliseconds_d_h);
        }

}


void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
    //free(compression_data->InputDataBlock);
    //free(compression_data->OutputDataBlock);
    //TODO FREE rest of data
    //free(compression_data);
    // Free CUDA memory
    cudaFree(compression_data->input_data);
    cudaFree(compression_data->output_data);
    cudaFree(compression_data->input_data_post_process);
    cudaFree(compression_data->missing_value);
    cudaFree(compression_data->missing_value_inverse);
    cudaFree(compression_data->zero_block_list);
    cudaFree(compression_data->zero_block_list_inverse);
    cudaFree(compression_data->compresion_identifier);
    cudaFree(compression_data->compresion_identifier_internal);
    cudaFree(compression_data->halved_samples);
    cudaFree(compression_data->size_block);
    cudaFree(compression_data->data_in_blocks);
    cudaFree(compression_data->compresion_identifier_best);
    cudaFree(compression_data->compresion_identifier_internal_best);
    cudaFree(compression_data->size_block_best);
    cudaFree(compression_data->bit_block_best);
    cudaFree(compression_data->data_in_blocks_best);

    // free CPU part
    free(compression_data->compresion_identifier_best_cpu);
    free(compression_data->compresion_identifier_best_internal_cpu);
    free(compression_data->size_block_best_cpu);
    free(compression_data->data_in_blocks_best_cpu);
    free(compression_data);





    
}