/**
 * \file device.c
 * \brief Benchmark #122 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */

#include "device.h"
#include "processing.h"
void syncstreams();

cudaStream_t cuda_streams[NUMBER_STREAMS];


void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	)
{
    init(compression_data,t, 0,0, device_name);
}



void init(
	compression_image_data_t *compression_data,
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
    t->start_dwt = new cudaEvent_t;
    t->stop_dwt = new cudaEvent_t;
    t->start_bpe = new cudaEvent_t;
    t->stop_bpe = new cudaEvent_t;
    t->start_memory_copy_device = new cudaEvent_t;
    t->stop_memory_copy_device = new cudaEvent_t;
    t->start_memory_copy_host = new cudaEvent_t;
    t->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(t->start_dwt);
    cudaEventCreate(t->stop_dwt);
    cudaEventCreate(t->start_bpe);
    cudaEventCreate(t->stop_bpe);
    cudaEventCreate(t->start_memory_copy_device);
    cudaEventCreate(t->stop_memory_copy_device);
    cudaEventCreate(t->start_memory_copy_host);
    cudaEventCreate(t->stop_memory_copy_host);

}


bool device_memory_init(
	compression_image_data_t *compression_data
	)
{
    // Allocate the device image imput
    cudaError_t err = cudaSuccess;
    // image process input is interger
    err = cudaMalloc((void **)&compression_data->input_image_gpu, sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }
    // input float image
   if (compression_data->type_of_compression)
    {
        // float opeation need to init the mid operation image
        err = cudaMalloc((void **)&compression_data->input_image_float, sizeof(float) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
        if (err != cudaSuccess)
        {
            return false;
        }
         // mid procesing float operation
         err = cudaMalloc((void **)&compression_data->transformed_float, sizeof(float) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
         if (err != cudaSuccess)
         {
             return false;
         }
          // Allocate the device low_filter
        err = cudaMalloc((void **)&compression_data->low_filter, LOWPASSFILTERSIZE * sizeof(float));

        if (err != cudaSuccess)
        {
            return false;
        }

        // Allocate the device high_filter
        err = cudaMalloc((void **)&compression_data->high_filter, HIGHPASSFILTERSIZE * sizeof(float));

        if (err != cudaSuccess)
        {
            return false;
        }
    }
    /*err = cudaMalloc((void **)&compression_data->output_image, sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }*/
    err = cudaMalloc((void **)&compression_data->transformed_image, sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }
    
    // Allocate the device image output in host memory
    compression_data->output_image = (int *)malloc(sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));



    for(unsigned int x = 0; x < NUMBER_STREAMS; ++x){
        cudaStreamCreate(&cuda_streams[x]);
    }
    
    return true;
}


void copy_memory_to_device(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	cudaEventRecord(*t->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(compression_data->input_image_gpu, compression_data->input_image, sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input_image from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    // float operation
    if (compression_data->type_of_compression)
    {
        

        err = cudaMemcpy(compression_data->low_filter, lowpass_filter_cpu, sizeof(float) * LOWPASSFILTERSIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector lowpass filter from host to device (error code %s)!\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpy(compression_data->high_filter, highpass_filter_cpu, sizeof(float) * HIGHPASSFILTERSIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector highpass filter from host to device (error code %s)!\n", cudaGetErrorString(err));
            return;
        }
	}
	cudaEventRecord(*t->stop_memory_copy_device);
}


void ccsds_wavelet_transform_2D(compression_image_data_t *device_object, unsigned int level)
{
    unsigned int h_size_level = device_object->h_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_w_lateral = ((device_object->w_size + device_object->pad_rows) / (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ
    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(size_w_lateral)/dimBlock.x));
    for(unsigned int i = 0; i < h_size_level; ++i)
    {
        if(device_object->type_of_compression)
        {
            
            // float
            wavelet_transform_float<<<dimGrid,dimBlock,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->input_image_float + ((device_object->w_size + device_object->pad_rows) * i), 
                device_object->transformed_float + ((device_object->w_size + device_object->pad_rows) * i), 
                size_w_lateral,
                device_object->low_filter,
                device_object->high_filter,
                1);
        }
        else
        {
            // integer

            wavelet_transform_int<<<dimGrid,dimBlock,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->input_image_gpu + ((device_object->w_size + device_object->pad_rows) * i), 
                device_object->transformed_image + ((device_object->w_size + device_object->pad_rows) * i), 
                size_w_lateral,
                1);
            wavelet_transform_low_int<<<dimGrid,dimBlock,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->input_image_gpu + ((device_object->w_size + device_object->pad_rows) * i), 
                device_object->transformed_image + ((device_object->w_size + device_object->pad_rows) * i), 
                size_w_lateral,
                1);
        }
        
    }
    // SYSC all threads
    syncstreams();

    // encode columns
    unsigned int w_size_level = device_object->w_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_h_lateral = ((device_object->h_size + device_object->pad_columns)/ (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ
    dim3 dimBlockColumn(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGridColumn(ceil(float(size_h_lateral)/dimBlockColumn.x));
    for(unsigned int i = 0; i < w_size_level; ++i)
    {
        
        if(device_object->type_of_compression)
        {
            wavelet_transform_float<<<dimGridColumn,dimBlockColumn,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object-> transformed_float+  i, 
                device_object->input_image_float + i, 
                size_h_lateral,
                device_object->low_filter,
                device_object->high_filter,
                device_object->w_size + device_object->pad_rows);
        
        }
        else
        {

            wavelet_transform_int<<<dimGridColumn,dimBlockColumn,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->transformed_image + i, 
                device_object->input_image_gpu + i, 
                size_h_lateral,
                device_object->w_size + device_object->pad_rows);

            wavelet_transform_low_int<<<dimGridColumn,dimBlockColumn,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->transformed_image + i, 
                device_object->input_image_gpu +  i, 
                size_h_lateral,
                device_object->w_size + device_object->pad_rows);
        }
    }
    // SYSC all threads
    syncstreams();
}


void process_benchmark(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{

	unsigned int h_size_padded = 0;
    unsigned int w_size_padded = 0;

   

	// create te new size
    h_size_padded = compression_data->h_size + compression_data->pad_rows;
    w_size_padded = compression_data->w_size + compression_data->pad_columns;
    unsigned int size = h_size_padded * w_size_padded;
    int  **transformed_image = NULL;
	transformed_image = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		transformed_image[i] = (int *)calloc(w_size_padded, sizeof(int));
	}

	unsigned int total_blocks =  (h_size_padded / BLOCKSIZEIMAGE )*(w_size_padded/ BLOCKSIZEIMAGE);
	int **block_string = NULL;
	block_string = (int **)calloc(total_blocks,sizeof(long *));
	for(unsigned int i = 0; i < total_blocks ; i++)
	{
		block_string[i] = (int *)calloc(BLOCKSIZEIMAGE * BLOCKSIZEIMAGE,sizeof(long));
	}
   
    // start computing the time
    cudaEventRecord(*t->start_dwt);
    // divide the flow depending of the type
    if(compression_data->type_of_compression)
    {
        // conversion to float
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(size)/dimBlock.x));
        transform_image_to_float<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(compression_data->input_image_gpu, compression_data->input_image_float, size);
        syncstreams();
    }
    // launch WAVELET 2D
    unsigned int iteration = 0;
    while(iteration != (LEVELS_DWT )){ 
        ccsds_wavelet_transform_2D(compression_data, iteration);
        ++iteration;
        
    }
    // finish the conversion 
    if(compression_data->type_of_compression)
    {
        // float opperation
        // transformation
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(size)/dimBlock.x));
        transform_image_to_int<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(compression_data->input_image_float, compression_data->input_image_gpu, size);
        syncstreams();
    }
    
    // copy the memory to device to compute the compression
    cudaMemcpy(compression_data->output_image, compression_data->input_image_gpu, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(*t->stop_dwt);
    T_START(t->t_dwt);

    

	// read the image data
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			transformed_image[i][j] = compression_data->output_image [i * h_size_padded + j];
		}
	}

    if (!compression_data->type_of_compression)
	{
        coefficient_scaling(transformed_image, h_size_padded, w_size_padded);
	}
	coeff_regroup(transformed_image, h_size_padded, w_size_padded);

    // create block string
    build_block_string(transformed_image, h_size_padded, w_size_padded,block_string);
	
	// calculate the number of segments that are generated by the block_string if have some left from the division we add 1 to the number of segments
	unsigned int num_segments = 0;
	if (total_blocks % compression_data->segment_size != 0)
	{
		num_segments = (total_blocks / compression_data->segment_size) + 1;
	}
	else
	{
		num_segments = total_blocks / compression_data->segment_size;
	}
	// create the array of SegmentBitStream
	// compute the BPE
	compression_data->segment_list = (SegmentBitStream *)malloc(sizeof(SegmentBitStream) * num_segments);
	// init the array of SegmentBitStream
	for (unsigned int i = 0; i < num_segments; ++i)
	{
		compression_data->segment_list[i].segment_id = 0;
		compression_data->segment_list[i].num_total_bytes = 0;
		compression_data->segment_list[i].first_byte = NULL;
		compression_data->segment_list[i].last_byte = NULL;
	}
	compression_data->number_of_segments = num_segments;
	compute_bpe(compression_data, block_string, num_segments);


	T_STOP(t->t_bpe);
	T_STOP(t->t_test);
	
	// clean image
	for(unsigned int i = 0; i < h_size_padded; i++){
			free(transformed_image[i]);
		}
	free(transformed_image);
	// free block_string
	for(unsigned int i = 0; i < total_blocks; i++){
			free(block_string[i]);
		}
	free(block_string);
}


void copy_memory_to_host(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	// empty
}


void get_elapsed_time(
	compression_image_data_t *compression_data, 
	compression_time_t *t,
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{

    float milliseconds_h_d = 0, milliseconds_d_h = 0;
    cudaEventElapsedTime(&milliseconds_h_d, *t->start_memory_copy_device, *t->stop_memory_copy_device);
    // kernel time 1
    long unsigned int application_miliseconds = (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *t->start_memory_copy_host, *t->stop_memory_copy_host);

	if (csv_format)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;\n", (float) milliseconds_h_d, elapsed_time, (float) milliseconds_d_h);
	}
	else if (database_format)
	{
		
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;%ld;\n", (float) milliseconds_h_d, elapsed_time, (float) milliseconds_d_h, timestamp);
	}
	else if(verbose_print)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("Elapsed time Host->Device: %.10f ms\n", (float) milliseconds_h_d);
		printf("Elapsed time kernel: %.10f ms\n", elapsed_time );
		printf("Elapsed time Device->Host: %.10f ms\n", (float) milliseconds_d_h);
	}
    
}



void clean(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	clean_segment_bit_stream(compression_data->segment_list, compression_data->number_of_segments);
	// clean the auxiliary structures
	free(compression_data->input_image);
	free(compression_data->segment_list);
    free(compression_data->output_image);
	free(compression_data);
}

void syncstreams(){
    for (unsigned int x = 0; x < NUMBER_STREAMS; ++x) {cudaStreamSynchronize (cuda_streams[x]);}
}