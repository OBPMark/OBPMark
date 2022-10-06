#include "../lib_functions.h"
//###############################################################################
//# Kernels
//###############################################################################


__global__ void
coeff_regroup(const int *A, int *B, const unsigned int h_size, const unsigned int w_size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i < h_size /8 && j < w_size/8)
    {
        // first band HH1 starts in i = (h_size>>1), j = (w_size>>1);
        const unsigned int i_hh =  (i * 4) + (h_size >>1);
        const unsigned int j_hh = (j * 4) + (w_size >>1);
        const unsigned int x_hh = ((i_hh - (h_size>>1)) << 1);
        const unsigned int y_hh = ((j_hh - (w_size>>1)) << 1);
        // second band HL1 starts in i = 0, j = (w_size>>1);
        const unsigned int i_hl =  i * 4;
        const unsigned int j_hl = (j * 4) + (w_size >>1);
        const unsigned int x_hl = (i_hl  << 1);
        const unsigned int y_hl = ((j_hl - (w_size>>1)) << 1);
        // third band LH1 starts in i = (h_size>>1), j = 0;
        const unsigned int i_lh =  (i * 4) + (h_size >>1);
        const unsigned int j_lh = j * 4;
        const unsigned int x_lh = ((i_lh - (h_size>>1)) << 1);
        const unsigned int y_lh = (j_lh<< 1);
        for (unsigned int p = 0; p < 4; ++p)
        {
            for (unsigned int k = 0; k < 4; ++k)
            {   
                B[(x_hh + (p+4)) * w_size + (y_hh + (k + 4))] = A[(i_hh + p - 4) * w_size + (j_hh + k -4)];
                B[(x_hl + p) * w_size + (y_hl + (k + 4))] = A[(i_hl + p) * w_size + (j_hl + k - 4)];
                B[(x_lh + (p+4)) * w_size + (y_lh + k)] = A[(i_lh + p - 4) * w_size + (j_lh + k)];
            }
        }
        // first band processed start second band
        // process hh2 band
        const unsigned int i_hh2 = (i * 2) + (h_size>>2);
        const unsigned int j_hh2 = (j * 2) + (w_size>>2);
        const unsigned int x_hh2 = ((i_hh2 - (h_size>>2)) <<2);
        const unsigned int y_hh2 = ((j_hh2 - (w_size>>2)) <<2);
        B[(x_hh2 + 2) * w_size + (y_hh2 + 2)] = A[(i_hh2) * w_size + (j_hh2)];
        B[(x_hh2 + 2) * w_size + (y_hh2 + 3)] = A[(i_hh2) * w_size + (j_hh2 +1)];
        B[(x_hh2 + 3) * w_size + (y_hh2 + 2)] = A[(i_hh2 + 1) * w_size + (j_hh2)];	
        B[(x_hh2 + 3) * w_size + (y_hh2 + 3)] = A[(i_hh2 + 1) * w_size + (j_hh2 + 1)];			
        
        // process hl2 band

        const unsigned int i_hl2 =  i * 2;
        const unsigned int j_hl2 = (j * 2) + (w_size>>2);
        const unsigned int x_hl2 = (i_hl2 <<2);
        const unsigned int y_hl2 = ((j_hl2 - (w_size>>2)) <<2);
        B[(x_hl2) * w_size + (y_hl2 + 2)] = A[(i_hl2) * w_size + (j_hl2)];	
        B[(x_hl2) * w_size + (y_hl2 + 3)] = A[(i_hl2) * w_size + (j_hl2 + 1)];	
        B[(x_hl2 + 1) * w_size + (y_hl2 + 2)] = A[(i_hl2 + 1) * w_size + (j_hl2)];		
        B[(x_hl2 + 1) * w_size + (y_hl2 + 3)] = A[(i_hl2 + 1) * w_size + (j_hl2 + 1)];				

        // process lh2 band
        const unsigned int i_lh2 =  (i * 2) + (h_size>>2);
        const unsigned int j_lh2 =  j * 2;
        const unsigned int x_lh2 = ((i_lh2 - (h_size>>2)) <<2);
        const unsigned int y_lh2 = (j_lh2<<2);
        B[(x_lh2 + 2) * w_size + (y_lh2)] = A[(i_lh2) * w_size + (j_lh2)];
        B[(x_lh2 + 2) * w_size + (y_lh2 + 1)] = A[(i_lh2) * w_size + (j_lh2 + 1)];			
        B[(x_lh2 + 3) * w_size + (y_lh2)] = A[(i_lh2+1) * w_size + (j_lh2)];			
        B[(x_lh2 + 3) * w_size + (y_lh2 + 1)] = A[(i_lh2 + 1) * w_size + (j_lh2 + 1)];

        // second band processed start thirt band
        const unsigned int x = (h_size>>3);
        // process hh3 band
        const unsigned int i_hh3 =  i + (h_size>>3);
        const unsigned int j_hh3 =  j + (w_size>>3);
        B[(((i_hh3 - x) <<3) + 1) * w_size + (((j_hh3 - (w_size>>3)) <<3) + 1)] = A[(i_hh3) * w_size + (j_hh3)];
        
        // process hl3 band
        const unsigned int i_hl3 =  i;
        const unsigned int j_hl3 =  j + (w_size>>3);
        B[(i_hl3 << 3) * w_size + (((j_hl3 - (w_size>>3)) <<3) + 1)] = A[(i_hl3) * w_size + (j_hl3)];
        
        // process lh3 band
        const unsigned int i_lh3 =  i + (h_size>>3);
        const unsigned int j_lh3 =  j;
        B[(((i_lh3 - x) <<3) + 1) * w_size + (j_lh3<<3)] = A[(i_lh3) * w_size + (j_lh3)];

        // process DC compoments
        B[(i<<3) * w_size + (j<<3)] = A[(i) * w_size + (j)];

    }
}

__global__ void
block_string_creation(const int *A, long *B, const unsigned int h_size, const unsigned int w_size)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < h_size/BLOCKSIZEIMAGE && j < w_size/BLOCKSIZEIMAGE)
    {
        for (unsigned int x = 0; x < BLOCKSIZEIMAGE; ++x)
            {
            for (unsigned int y =0; y < BLOCKSIZEIMAGE; ++y)
            {
                B[(i + j) * w_size + (x * BLOCKSIZEIMAGE + y)] = long(A[(i*BLOCKSIZEIMAGE +x) * w_size + (j*BLOCKSIZEIMAGE+y)]);
            }
        }
    }
}

__global__ void
transform_image_to_float(const int *A, float *B, unsigned int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < size)
    {
        B[i] = float(A[i]);
    }
    
}

__global__ void
transform_image_to_int(const float *A, int *B, unsigned int size)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < size)
    {
        B[i] = A[i] >= 0 ? int(A[i] + 0.5) : int(A[i] - 0.5);
    }
    
}


__global__ void
wavelet_transform_low_int(const int *A, int *B, const int n, const int step){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        int sum_value_low = 0;
        if(i == 0){
            sum_value_low = A[0] - floor(- (B[(size * step)]/2.0) + (1.0/2.0));
        }
        else
        {
            sum_value_low = A[(2 * i) * step] - floor( - (( B[(i * step) + (size * step) -(1 * step)] +  B[(i * step) + (size*step)])/ 4.0) + (1.0/2.0) );
        }
        
        B[(i * step)] = sum_value_low;
    }
}
__global__ void
wavelet_transform_int(const int *A, int *B, const int n, const int step){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (i < size){
        int sum_value_high = 0;
        // specific cases
        if(i == 0){
            sum_value_high = A[1 * step] - floor( ((9.0/16.0) * (A[0] + A[2* step])) - ((1.0/16.0) * (A[2* step] + A[4* step])) + (1.0/2.0));
        }
        else if(i == size -2){
            sum_value_high = A[ (2*size  - 3) * step] - floor( ((9.0/16.0) * (A[(2*size -4) * step] + A[(2*size -2)*step])) - ((1.0/16.0) * (A[(2*size - 6)* step] + A[(2*size - 2) * step])) + (1.0/2.0));
        }
        else if(i == size - 1){
            sum_value_high = A[(2*size - 1)* step] - floor( ((9.0/8.0) * (A[(2*size  -2) * step])) -  ((1.0/8.0) * (A[(2*size  - 4)* step ])) + (1.0/2.0));
        }
        else{
            // generic case
            sum_value_high = A[(2*i  +1)* step] - floor( ((9.0/16.0) * (A[(2*i)* step ] + A[(2*i +2)* step])) - ((1.0/16.0) * (A[(2*i  - 2)* step] + A[(2*i  + 4)* step])) + (1.0/2.0));
        }
        
        //store
        B[(i * step)+(size * step)] = sum_value_high;

        //__syncthreads();
        // low_part
        //for (unsigned int i = 0; i < size; ++i){
        
        //}
    }

}

__global__ void
wavelet_transform_float(const float *A, float *B, const int n, const float *lowpass_filter,const float *highpass_filter, const int step){
    unsigned int size = n;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    int gi_end = HIGHPASSFILTERSIZE / 2;
    if (i < size){
        float sum_value_low = 0;
        for (int hi = hi_start; hi < hi_end + 1; ++hi){
			int x_position = (2 * i) + hi;
			if (x_position < 0) {
				// turn negative to positive
				x_position = x_position * -1;
			}
			else if (x_position > full_size - 1)
			{
				x_position = full_size - 1 - (x_position - (full_size -1 ));;
			}
			// now I need to restore the hi value to work with the array
			sum_value_low += lowpass_filter[hi + hi_end] * A[x_position * step];
			
        }
		// store the value
		B[i * step] = sum_value_low;
		float sum_value_high = 0;
		// second process the Highpass filter
		for (int gi = gi_start; gi < gi_end + 1; ++gi){
			int x_position = (2 * i) + gi + 1;
			if (x_position < 0) {
				// turn negative to positive
				x_position = x_position * -1;
			}
			else if (x_position >  full_size - 1)
			{
				x_position = full_size - 1 - (x_position - (full_size -1 ));
			}
			sum_value_high += highpass_filter[gi + gi_end] * A[x_position * step];
		}
		// store the value
		B[(i * step) + (size * step)] = sum_value_high;

    }
}


//###############################################################################
cudaStream_t cuda_streams[NUMBER_STREAMS];

int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns);
void ccsds_wavelet_transform_2D(DataObject *device_object, unsigned int level);
void header_inilization(header_struct *header_pointer, BOOL DWT_type, unsigned int pad_rows, unsigned int image_width, BOOL first, BOOL last);
void encode_dc(block_struct* list_of_process_data, long *block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks,unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer);
void dpcm_dc_mapper(unsigned long *shifted_dc, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_entropy_encoder(block_struct* list_of_process_data, unsigned int block_position, unsigned short *dc_remainer, unsigned char quantization_factor, header_struct *header_pointer, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_encoder(unsigned int gaggle_id, unsigned int block_position, block_struct* list_of_process_data, header_struct *header_pointer, unsigned long *dc_mapper, short N, int star_index, int gaggles, int max_k, int id_length);
void convert_to_bits_segment(FILE *ofp, block_struct segment_data);
unsigned long conversion_twos_complement(long original, short leftmost);

void syncstreams();


void init(DataObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}

void init(DataObject *device_object, int platform ,int device, char* device_name){
	cudaSetDevice(device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	//printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start_dwt = new cudaEvent_t;
    device_object->stop_dwt = new cudaEvent_t;
    device_object->start_bpe = new cudaEvent_t;
    device_object->stop_bpe = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start_dwt);
    cudaEventCreate(device_object->stop_dwt);
    cudaEventCreate(device_object->start_bpe);
    cudaEventCreate(device_object->stop_bpe);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}

bool device_memory_init(DataObject *device_object){
    // Allocate the device image imput
    cudaError_t err = cudaSuccess;
    // image process input is interger
    err = cudaMalloc((void **)&device_object->input_image, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }
    // input float image
    if (device_object->type)
    {
        // float opeation need to init the mid operation image
        err = cudaMalloc((void **)&device_object->input_image_float, sizeof(float) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
        if (err != cudaSuccess)
        {
            return false;
        }

         // mid procesing float operation
         err = cudaMalloc((void **)&device_object->transformed_float, sizeof(float) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
         if (err != cudaSuccess)
         {
             return false;
         }
          // Allocate the device low_filter
        err = cudaMalloc((void **)&device_object->low_filter, LOWPASSFILTERSIZE * sizeof(float));

        if (err != cudaSuccess)
        {
            return false;
        }

        // Allocate the device high_filter
        err = cudaMalloc((void **)&device_object->high_filter, HIGHPASSFILTERSIZE * sizeof(float));

        if (err != cudaSuccess)
        {
            return false;
        }
    }
    err = cudaMalloc((void **)&device_object->output_image, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }
    
    // output_image
    err = cudaMalloc((void **)&device_object->transformed_image, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->final_transformed_image, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->coeff_image_regroup, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    if (err != cudaSuccess)
    {
        return false;
    }

	unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
    unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
    err = cudaMalloc((void **)&device_object->block_string, sizeof(long) * block_string_size);
    if (err != cudaSuccess)
    {
        return false;
    }
    


    for(unsigned int x = 0; x < NUMBER_STREAMS; ++x){
        cudaStreamCreate(&cuda_streams[x]);
    }
    
    return true;


}

void copy_data_to_gpu(DataObject *device_object, int* image_data)
{
    cudaEventRecord(*device_object->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(device_object->input_image, image_data, sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input_image from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    // float operation
    if (device_object->type)
    {
        err = cudaMemcpy(device_object->low_filter, lowpass_filter_cpu, sizeof(float) * LOWPASSFILTERSIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector lowpass filter from host to device (error code %s)!\n", cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpy(device_object->high_filter, highpass_filter_cpu, sizeof(float) * HIGHPASSFILTERSIZE, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector highpass filter from host to device (error code %s)!\n", cudaGetErrorString(err));
            return;
        }
	}
	cudaEventRecord(*device_object->stop_memory_copy_device);
}

void copy_data_to_cpu(DataObject *device_object, long* block_string)
{   
	unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
	unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
	
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(block_string, device_object->block_string, block_string_size * sizeof(long), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
   
}

void encode_engine(DataObject *device_object, int* image_data)
{
    unsigned int h_size_padded = 0;
    unsigned int w_size_padded = 0;

	// create te new size
    h_size_padded = device_object->h_size + device_object->pad_rows;
    w_size_padded = device_object->w_size + device_object->pad_columns;
    unsigned int size = h_size_padded * w_size_padded;
    //int* image_data = readBMP(device_object->filename_input, device_object->pad_rows, device_object->pad_columns);
    // start computing the time
    copy_data_to_gpu(device_object, image_data);
    cudaEventRecord(*device_object->start_dwt);
    // divide the flow depending of the type
    if(device_object->type)
    {
        // conversion to float
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(size)/dimBlock.x));
        transform_image_to_float<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(device_object->input_image, device_object->input_image_float, size);
        syncstreams();
    }
    // launch WAVELET 2D
    unsigned int iteration = 0;
    while(iteration != (LEVELS_DWT )){ 
        ccsds_wavelet_transform_2D(device_object, iteration);
        ++iteration;
    }
    // finish the conversion 
    if(device_object->type)
    {
        // float opperation
        // transformation
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(size)/dimBlock.x));
        transform_image_to_int<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(device_object->input_image_float, device_object->output_image, size);
        syncstreams();
    }
    else
    {
        // integer operation
        // copy the memory
        cudaMemcpy(device_object->output_image, device_object->input_image, sizeof(int) * size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(*device_object->stop_dwt);
    // FINSH DWT 2D
    cudaEventRecord(*device_object->start_bpe);
    // coeff_regroup
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid = dim3(ceil(float(h_size_padded/8)/dimBlock.x), ceil(float(w_size_padded/8)/dimBlock.y));
    coeff_regroup<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(device_object->output_image, device_object->coeff_image_regroup, h_size_padded, w_size_padded);

    // block string creation
    unsigned int block_h = h_size_padded / BLOCKSIZEIMAGE;
	unsigned int block_w = w_size_padded / BLOCKSIZEIMAGE;
    dimGrid = dim3(ceil(float(block_h)/dimBlock.x), ceil(float(block_w)/dimBlock.y));
    block_string_creation<<<dimGrid,dimBlock,0, cuda_streams[0]>>>(device_object->coeff_image_regroup, device_object->block_string, h_size_padded, w_size_padded);
    syncstreams();
    cudaEventRecord(*device_object->stop_bpe);
    // accelerated GPU procesing finish
    // start secuential processing
    unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
    unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
    
	long *block_string = (long *)malloc(sizeof(long) * block_string_size);
    copy_data_to_cpu(device_object, block_string);
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_bpe_cpu);
	// loop over all blocks in Segment size portions
	BOOL DWT_type = device_object->type ? TRUE : FALSE;
	// create the list of data to print
	unsigned int segment = 0;
	unsigned int number_of_segment = total_blocks / SEGMENTSIZE;
	block_struct *list_of_process_data = NULL;
	list_of_process_data = (block_struct *) calloc(number_of_segment, sizeof(block_struct)); 
	for (unsigned int block_counter = 0; block_counter < total_blocks; block_counter += SEGMENTSIZE)
	{
		// Step 3 create the header data
		// create the header
        header_struct *ptr_header = (header_struct *)malloc(sizeof(header_struct));
		// inicialize header
		
		if (block_counter == 0){
			// first block
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, TRUE, FALSE);
		}
		else if (block_counter + SEGMENTSIZE == total_blocks)
		{
			// last block
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, TRUE);
		}
		else if (block_counter + SEGMENTSIZE > total_blocks)
		{			
			// Special sitiuation when the number of blocks per segment are not the same
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, TRUE);
			ptr_header->header.part1.part_3_flag = TRUE;
			ptr_header->header.part3.seg_size_blocks_20bits = total_blocks - block_counter;

			ptr_header->header.part1.part_2_flag = TRUE;
			ptr_header->header.part2.seg_byte_limit_27bits = BITSPERFIXEL * (total_blocks - block_counter) * 64/8;
		}
		else
		{
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, FALSE);
		}
		// Step 4 encode DC
		// array that stores the maximun bit in AC in each of the blocks in a segment
		unsigned short *bit_max_in_segment = NULL;
		// get the numer of blocks in this segment
		unsigned int blocks_in_segment = block_counter + SEGMENTSIZE > total_blocks ? total_blocks - block_counter : SEGMENTSIZE ;
		bit_max_in_segment = (unsigned short*)calloc(blocks_in_segment, sizeof(unsigned short));
		unsigned char quantization_factor = 0;
		unsigned short *dc_remainer = NULL;
        dc_remainer = (unsigned short*)calloc(blocks_in_segment, sizeof(unsigned short));
		encode_dc(list_of_process_data, block_string, ptr_header, block_counter, total_blocks,bit_max_in_segment, &quantization_factor,dc_remainer);
		// copy data
		list_of_process_data[segment].dc_remainer = dc_remainer;
		list_of_process_data[segment].quantization_factor = quantization_factor;
		list_of_process_data[segment].header = (header_struct_base *) calloc( 1,sizeof(header_struct_base));
		list_of_process_data[segment].header->part1 = ptr_header->header.part1;
		list_of_process_data[segment].header->part2 = ptr_header->header.part2;
		list_of_process_data[segment].header->part3 = ptr_header->header.part3;
		list_of_process_data[segment].header->part4 = ptr_header->header.part4;
		++segment;
		//free(ptr_header);

	}
	// end processing
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_bpe_cpu);
    // store the data
	FILE *ofp;
    ofp = fopen(device_object->filename_output, "w");

    if (ofp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", device_object->filename_output);
    exit(1);
    }
	// loop over the data
	for (unsigned int i = 0; i < number_of_segment; ++i)
	{
		convert_to_bits_segment(ofp, list_of_process_data[i]);
	}

	fclose(ofp);

}

/*int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns)
{
	BMP Image;
   	Image.ReadFromFile( filename );
    int size_output =  Image.TellWidth() * Image.TellHeight();
	int  *data_bw = NULL;
	data_bw = (int*)calloc((Image.TellHeight() + pad_rows ) * (Image.TellWidth() + pad_columns), sizeof(int));
   	// convert each pixel to greyscale
   	for( int i=0 ; i < Image.TellHeight() ; i++)
   	{
    	for( int j=0 ; j < Image.TellWidth() ; j++)
    	{
			data_bw[i * Image.TellWidth() + j] =  (Image(j,i)->Red + Image(j,i)->Green + Image(j,i)->Blue)/3;
    	}
   }
   //we need to duplicate rows and columns to be in BLOCKSIZEIMAGE
   for(unsigned int i = 0; i < pad_rows ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_columns; j++)
			data_bw[(i + Image.TellHeight()) * Image.TellWidth() + j] = data_bw[(Image.TellHeight() - 1)* Image.TellWidth() + j];
	}

	for(unsigned int i = 0; i < pad_columns ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_rows ; j++)
			data_bw[(j)* Image.TellWidth() + (i + Image.TellWidth())] = data_bw[j * Image.TellWidth() + ( Image.TellWidth() - 1)];
	}
    
   return data_bw;
}*/

void ccsds_wavelet_transform_2D(DataObject *device_object, unsigned int level)
{
    unsigned int h_size_level = device_object->h_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_w_lateral = ((device_object->w_size + device_object->pad_rows) / (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ
    dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
    dim3 dimGrid(ceil(float(size_w_lateral)/dimBlock.x));
    for(unsigned int i = 0; i < h_size_level; ++i)
    {
        if(device_object->type)
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
            wavelet_transform_low_int<<<dimGrid,dimBlock,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->input_image + ((device_object->w_size + device_object->pad_rows) * i), 
                device_object->transformed_image + ((device_object->w_size + device_object->pad_rows) * i), 
                size_w_lateral,
                1);
            wavelet_transform_int<<<dimGrid,dimBlock,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->input_image + ((device_object->w_size + device_object->pad_rows) * i), 
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
        
        if(device_object->type)
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
            wavelet_transform_low_int<<<dimGridColumn,dimBlockColumn,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->transformed_image + i, 
                device_object->input_image +  i, 
                size_h_lateral,
                device_object->w_size + device_object->pad_rows);
            wavelet_transform_int<<<dimGridColumn,dimBlockColumn,0,cuda_streams[i % NUMBER_STREAMS]>>>
            (device_object->transformed_image + i, 
                device_object->input_image + i, 
                size_h_lateral,
                device_object->w_size + device_object->pad_rows);
        }
    }
    // SYSC all threads
    syncstreams();
}

void syncstreams(){
    for (unsigned int x = 0; x < NUMBER_STREAMS; ++x) {cudaStreamSynchronize (cuda_streams[x]);}
}

void get_elapsed_time(DataObject *device_object, bool csv_format){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0, miliseconds_bpe = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time 1
    cudaEventElapsedTime(&milliseconds, *device_object->start_dwt, *device_object->stop_dwt);
    // kernel time 2
    cudaEventElapsedTime(&miliseconds_bpe, *device_object->start_bpe, *device_object->stop_bpe);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    // part 2 of the BPE
    miliseconds_bpe += (device_object->end_bpe_cpu.tv_sec - device_object->start_bpe_cpu.tv_sec) * 1000 + (device_object->end_bpe_cpu.tv_nsec - device_object->start_bpe_cpu.tv_nsec) / 1000000;

    if (csv_format){
         printf("%.10f;%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,miliseconds_bpe,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel DWT: %.10f miliseconds\n", milliseconds);
         printf("Elapsed time kernel BPE: %.10f miliseconds\n", miliseconds_bpe);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
}

//encoder part
void header_inilization(header_struct *header_pointer, BOOL DWT_type ,unsigned int pad_rows, unsigned int image_width, BOOL first, BOOL last)
{
	header_pointer->header.part1.start_img_flag = first;
	header_pointer->header.part1.end_img_flag = last;
	header_pointer->header.part1.segment_count_8bits = 0;
	header_pointer->header.part1.bit_depth_ac_5bits = 0;
	header_pointer->header.part1.bit_depth_dc_5bits = 0;
	header_pointer->header.part1.part_2_flag = TRUE;
	header_pointer->header.part1.part_3_flag = TRUE;
	header_pointer->header.part1.part_4_flag = TRUE;
	header_pointer->header.part1.pad_rows_3bits = pad_rows;
	header_pointer->header.part1.reserved_5bits = 0;

	header_pointer->header.part2.seg_byte_limit_27bits = 0; 
	header_pointer->header.part2.dc_stop = FALSE;
	header_pointer->header.part2.bit_plane_stop_5bits = 0;
	header_pointer->header.part2.stage_stop_2bits = 3; // 2 bits, transform input data quantization. TODO chech if should be 0
	header_pointer->header.part2.use_fill = header_pointer->header.part2.seg_byte_limit_27bits == 0 ? FALSE : TRUE;
	header_pointer->header.part2.reserved_4bits = 0;

	header_pointer->header.part3.seg_size_blocks_20bits = SEGMENTSIZE;
	header_pointer->header.part3.opt_dc_select = TRUE;
	header_pointer->header.part3.opt_ac_select = TRUE;
	header_pointer->header.part3.reserved_2bits = 0;

	header_pointer->header.part4.dwt_type = TRUE;
	header_pointer->header.part4.reserved_2bits = 0; 
	header_pointer->header.part4.signed_pixels = FALSE;
	header_pointer->header.part4.pixel_bit_depth_4bits = 8;
	header_pointer->header.part4.image_with_20bits = image_width;
	header_pointer->header.part4.transpose_img = FALSE;
	header_pointer->header.part4.code_word_length = 000;
	header_pointer->header.part4.custom_wt_flag = FALSE;
	header_pointer->header.part4.custom_wt_HH1_2bits = 0;
	header_pointer->header.part4.custom_wt_HL1_2bits = 0;
	header_pointer->header.part4.custom_wt_LH1_2bits = 0;
	header_pointer->header.part4.custom_wt_HH2_2bits = 0;
	header_pointer->header.part4.custom_wt_HL2_2bits = 0;
	header_pointer->header.part4.custom_wt_LH2_2bits = 0;
	header_pointer->header.part4.custom_wt_HH3_2bits = 0;
	header_pointer->header.part4.custom_wt_HL3_2bits = 0;
	header_pointer->header.part4.custom_wt_LH3_2bits = 0;
	header_pointer->header.part4.custom_wt_LL3_2bits = 0;
	header_pointer->header.part4.reserved_11bits = 0;
}

void encode_dc(block_struct* list_of_process_data, long *block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks, unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer)
{
	long  max_dc_per_segment = 0;

	signed long dc_min = 0x10000;
	signed long dc_max = -0x10000;

	signed long max_ac_segment = 0;

	// Start of the calculations to get the bit depth in the segment in the DC range and AC
	unsigned int blocks_in_segment = block_position + SEGMENTSIZE > total_blocks ? total_blocks - block_position : SEGMENTSIZE ;
	// loop over a segment щ（ﾟДﾟщ）
	// determine max position
	unsigned int max_position = block_position + SEGMENTSIZE > total_blocks ? total_blocks : block_position + SEGMENTSIZE; // min operation (づ｡◕‿‿◕｡)づ
	// one poisition stores a segment so we loop over all of the DC values
	// this loop is for extract all of the maximun values of the DC operands and AC
	for (unsigned int block_position_in_segment = block_position; block_position_in_segment < max_position; ++block_position_in_segment)
	{
        if(TOPOSITIVE(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0]) > TOPOSITIVE(max_dc_per_segment))
		{
            // update the value of max dc in this segment
		    max_dc_per_segment = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0];
		}
		
		
		if(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0] > dc_max)
		{
			dc_max = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0];
		}

		if(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0] > dc_min)
		{
			dc_min = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0];
		}

		signed long ac_bit_max_one_block = 0; 
		for(unsigned int i = 1; i < BLOCKSIZEIMAGE * BLOCKSIZEIMAGE; ++i)
		{
			unsigned int abs_ac = 0;
			abs_ac = TOPOSITIVE(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +i]);
			if ((signed long)abs_ac > ac_bit_max_one_block)
			{
				ac_bit_max_one_block = abs_ac;
			}
			if ((signed long)abs_ac > max_ac_segment)
			{
				max_ac_segment = abs_ac;
			}
		}
		// extrat the numer of bit maximun per segment for high AC segment (╯°□°）╯︵ ┻━┻ 

		while (ac_bit_max_one_block > 0)
		{
			ac_bit_max_one_block >>= 1; // divide by two
			++bit_max_in_segment[block_position_in_segment - block_position];
		}
    }
	// finish the part of extracting data from all of the blocks in a segment
	header_pointer->header.part1.bit_depth_ac_5bits = 0;
	header_pointer->header.part1.bit_depth_dc_5bits = 0;
	// get the maximun bit depth in ac in all of the blocks
	while (max_ac_segment > 0)
	{
		max_ac_segment >>= 1;
		++header_pointer->header.part1.bit_depth_ac_5bits;
	}
	// get the maximuin DC depth in all blocks
	if (dc_min >= 0)
	{
		max_dc_per_segment = dc_max;
	}
	else if (dc_max <= 0)
	{
		max_dc_per_segment = dc_min;
	}
	else if (dc_max >= TOPOSITIVE(dc_min))
	{
		max_dc_per_segment = dc_max;
	}
	else
	{
		max_dc_per_segment = dc_min;
	}
	
	if (max_dc_per_segment >=0)
	{
		while (max_dc_per_segment > 0)
		{
			max_dc_per_segment >>=1; // divide by two (づ｡◕‿‿◕｡)づ
			++header_pointer->header.part1.bit_depth_dc_5bits;
		}
		
	}
	else
	{
		unsigned long temp = -max_dc_per_segment; // chaneg sing of DC in other to extra the numer of bits
		while (temp > 0)
		{
			temp >>=1; // divide by two (づ｡◕‿‿◕｡)づ
			++header_pointer->header.part1.bit_depth_dc_5bits;
		}
		if( (1 << (header_pointer->header.part1.bit_depth_dc_5bits -1)) == - max_dc_per_segment)
		{
			--header_pointer->header.part1.bit_depth_dc_5bits;
		}
	}
	// include the sign bit
	++header_pointer->header.part1.bit_depth_dc_5bits;

	// END of the calculations to get the bit depth in the segment in the DC range and AC
	// all this for get the bit_depth (╯°□°）╯︵ ┻━┻ 

	// copy the values to variables
	unsigned short bit_depth_dc = header_pointer->header.part1.bit_depth_dc_5bits;
	unsigned short bit_depth_ac = header_pointer->header.part1.bit_depth_ac_5bits;
	
	// convert the header to binary
	// TODO

	// start with the DC conversion ¯\\_(ツ)_/¯  I hoppe
	//////////////// Start of the table 4-8. Pages 44,45 CCSDS 122 ///////////////
	unsigned short quantization_of_factor_q_prime= 0;
	if (bit_depth_dc <= 3)
	{
		quantization_of_factor_q_prime = 0;
	}
	else if (((bit_depth_dc - (1 + (bit_depth_ac >> 1))) <= 1 ) && (bit_depth_dc > 3))
	{
		quantization_of_factor_q_prime = bit_depth_dc - 3;
	}
	else if (((bit_depth_dc - (1 + (bit_depth_ac >> 1))) > 10 ) && (bit_depth_dc > 3))
	{
		quantization_of_factor_q_prime = bit_depth_dc - 10;
	}
	else
	{
		quantization_of_factor_q_prime = 1 + (bit_depth_ac>>1);
	}
	//////////////// End of the table 4-8. Pages 44,45 CCSDS 122 ///////////////
	// Point 4.3.1.3 
	*quantization_factor = quantization_of_factor_q_prime > header_pointer->header.part4.custom_wt_LL3_2bits ? quantization_of_factor_q_prime : header_pointer->header.part4.custom_wt_LL3_2bits; // max operation (づ｡◕‿‿◕｡)づ


	// Shift of the DC component
	// k consists of quantization_factor ones in the least significant bits ¯\\_(ツ)_/¯ i suposse
	
	unsigned int k = (1 << *quantization_factor)-1;
	
	// shift the DC component to the right
	unsigned long *shifted_dc = NULL;
	shifted_dc = (unsigned long*)calloc(blocks_in_segment, sizeof(unsigned long));
	
	for(unsigned int i = 0; i < blocks_in_segment; ++i)
	{
		// convbert the DC value to twos complement
		unsigned long new_value = conversion_twos_complement (block_string[i * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0], bit_depth_dc);
		shifted_dc[i] = (new_value >> *quantization_factor);
		dc_remainer[i] = (unsigned short) (new_value & k); // Binary AND ¯\\_(ツ)_/¯ 
	}
	
	unsigned char N = header_pointer->header.part1.bit_depth_dc_5bits - *quantization_factor > 1 ? header_pointer->header.part1.bit_depth_dc_5bits - *quantization_factor: 1; // max operation (づ｡◕‿‿◕｡)づ
	//The DC coefficient quantization rules in table 4-8 limit the value of N to be
    //within 10, and thus coding options are defined only up to N =10.||  4.3.2.1

	//When N is 1, each quantized DC coefficient c'm consists of a single bit. In this case,
	//the coded quantized DC coefficients for a segment consist of these bits, concatenated
	//together. Otherwise N>1 and the quantized DC coefficients in a segment, c'm, shall be
	//encoded using the procedure described below.  || 4.3.2.2
	//The first quantized DC coefficient for every sequence of S consecutive coefficients,
	//referred to as a reference sample, shall be written to the encoded bitstream directly (i.e.,
	//without any further processing or encoding) || 4.3.2.3
	list_of_process_data[block_position/SEGMENTSIZE].N = N;
	list_of_process_data[block_position/SEGMENTSIZE].shifted_dc_1bit = shifted_dc;
	if (N == 1)
	{

	}
	else
	{
		// Sample-spit entropy for bit shifted DC's || 4.3.2.6
		unsigned long *dc_mapper = NULL;
		dc_mapper = (unsigned long*)calloc(blocks_in_segment, sizeof(unsigned long));
		dpcm_dc_mapper(shifted_dc,dc_mapper,blocks_in_segment,N);
	
		dc_entropy_encoder(list_of_process_data,block_position ,dc_remainer,*quantization_factor,header_pointer,dc_mapper,blocks_in_segment, N);
		list_of_process_data[block_position/SEGMENTSIZE].dc_mapper = dc_mapper;
	}

	
}

void dpcm_dc_mapper(unsigned long *shifted_dc, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N)
{
	long theta = 0;
	unsigned long bits1 = 0; 
	long x_min = 0;
	long x_max = 0;

	long * dc_diff = NULL; 
	dc_diff = (long *)calloc(blocks_in_segment,sizeof(long));
	// change the first position of the shifted_dc to mapp that is 0
	dc_mapper[0] = shifted_dc[0];
	for (unsigned int i = 0; i < (N -1); ++i)
	{
		bits1 = (bits1 <<1)+1;
	}

	if((shifted_dc[0] & (1 << (N -1))) > 0) // minus 0
	{
		shifted_dc[0] = -(short)( ((shifted_dc[0]^bits1) & bits1) + 1 ); // negative 
	}
	else
	{ 
		shifted_dc[0] =  shifted_dc[0];// positive. 
	}

	dc_diff[0] = shifted_dc[0];
	for (unsigned int i = 1; i < blocks_in_segment; ++i)
	{
		if((shifted_dc[i] & (1 << (N -1))) > 0) // minus 0
		{
			shifted_dc[i] = -(short)( ((shifted_dc[i]^bits1) & bits1) + 1 );
		}
		else
		{
			shifted_dc[0] =  shifted_dc[0];// positive.
		}
		// get the diff
		dc_diff[i] = shifted_dc[i] - shifted_dc[i - 1];
	}

	for (unsigned int i = 1; i <blocks_in_segment; ++i) // 4.3.2.4 equations 18 and 19
	{
		theta = shifted_dc[i - 1] - x_min > x_max - shifted_dc[i - 1] ? x_max - shifted_dc[i - 1] : shifted_dc[i - 1] - x_min; // min operation (づ｡◕‿‿◕｡)づ
		if(dc_diff[i] >= 0 && dc_diff[i] <= theta)
		{
			dc_mapper[i] = 2*dc_diff[i];
		}
		else if (dc_diff[i] < 0 && dc_diff[i] >= -theta)
		{
			dc_mapper[i] = 2 * abs(dc_diff[i]) -1;
		}
		else
		{
			dc_mapper[i] = theta + abs(dc_diff[i]);
		}
	
	}
	free(dc_diff);
}

void dc_entropy_encoder(block_struct* list_of_process_data, unsigned int block_position, unsigned short *dc_remainer, unsigned char quantization_factor, header_struct *header_pointer, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N)
{
	int max_k = 0;
	int id_length = 0;


	// CODE Option identifiers Table 4-9 4.3.2.7
	if (N == 2)
	{
		max_k = 0;
		id_length = 1;
	}
	else if (N <= 4)
	{
		max_k = 2;
		id_length = 2;
	}
	else if ( N <= 8)
	{
		max_k = 6;
		id_length = 3;
	}
	else if (N <= 10)
	{
		max_k = 8;
		id_length = 4;
	}
	
	// Gaggle processing
	unsigned int gaggle_index = 0;
	unsigned int gaggles = 0;
	unsigned int total_gaggles = blocks_in_segment / GAGGLE_SIZE;
	unsigned int gaggle_id = 0;
	list_of_process_data[block_position/SEGMENTSIZE].bits_min_k = id_length;
	list_of_process_data[block_position/SEGMENTSIZE].min_k = (int *)calloc(total_gaggles,sizeof(int));

	while(gaggle_index < blocks_in_segment)
	{
		gaggles = GAGGLE_SIZE > (blocks_in_segment - gaggle_index) ? (blocks_in_segment - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
		dc_encoder(gaggle_id, block_position, list_of_process_data, header_pointer, dc_mapper, N, gaggle_index,gaggles, max_k, id_length);
		gaggle_index += gaggles;
		++gaggle_id;
	}
	// Additional bit planes opf DC coefficients 4.3.3
	if (header_pointer->header.part1.bit_depth_ac_5bits < quantization_factor)
	{
		int num_add_bit_planes;
		if (header_pointer->header.part4.dwt_type == INTEGER_WAVELET)
		{
			num_add_bit_planes = quantization_factor - (header_pointer ->header.part1.bit_depth_ac_5bits > header_pointer->header.part4.custom_wt_LL3_2bits ? header_pointer ->header.part1.bit_depth_ac_5bits : header_pointer->header.part4.custom_wt_LL3_2bits); // max operation (づ｡◕‿‿◕｡)づ
		}
		else
		{
			num_add_bit_planes = quantization_factor - header_pointer->header.part1.bit_depth_ac_5bits;
		}
		list_of_process_data[block_position/SEGMENTSIZE].numaddbitplanes = num_add_bit_planes;
		list_of_process_data[block_position/SEGMENTSIZE].dc_remainer = dc_remainer;
		// loop over the remainders to print

	}
	

}


void dc_encoder(unsigned int gaggle_id, unsigned int block_position,block_struct* list_of_process_data, header_struct *header_pointer, unsigned long *dc_mapper, short N, int star_index, int gaggles, int max_k, int id_length)
{
	
	int min_k = 0;
	unsigned long min_bits = 0xFFFF;
	unsigned long total_bits = 0;

	unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
	// determine code choice min_k to use, either via brute-force optimum calculation, or heuristic approach
	// 4.3.2.11
	if (header_pointer->header.part3.opt_dc_select == TRUE) // Brute force (╯°□°）╯︵ ┻━┻ 
	{
		min_k = uncoded_flag;  // uncoded option used, unless we find a better option below
		for(unsigned int k = 0; k <= max_k; ++k)
		{
			if (star_index == 0)
			{
				total_bits = N;
			}
			else
			{
				total_bits = 0;
			}
			int max_value = star_index > 1 ? star_index : 1; // max operation (づ｡◕‿‿◕｡)づ
			for (unsigned int i = max_value; i < star_index + gaggles; ++i)
			{
				total_bits += ((dc_mapper[i] >> k) + 1) + k; // coded sample cost
			}
			if((total_bits < min_bits) && (total_bits < (N * gaggles)))
			{
				min_bits = total_bits;
				min_k = k;
			}
		}
		
	}
	else // heuristic code option selection (っ▀¯▀)つ
	{
		int delta = 0;
		int j = gaggles;
		if (star_index == 0)
		{
			j = gaggles -1;
		}
		for (unsigned int i = star_index; i < star_index + gaggles; ++i)
		{
			delta += dc_mapper[i];
		}
		// Table 4-10
		if (64*delta >= 23*j*(1 << N))
		{
			min_k = uncoded_flag; // indicate that uncoded option is to be used
		}
		else if (207 * j > 128 * delta)
		{
			min_k = 0;
		}
		else if ((long)(j * (1<< (N + 5)))  <= (long)(128 * delta + 49 * j))
		{
			min_k = N -2;
		}
		else
		{
			min_k = 0;
			// ¯\\_(ツ)_/¯ 
			while ((long)(j * (1 << (min_k + 7))) <= (long)(128 - delta + 49 *j) )
			{
				++min_k; // ¯\\_(ツ)_/¯ 
			}
			--min_k; // ¯\\_(ツ)_/¯ 
			
		}
	}
	list_of_process_data[block_position/SEGMENTSIZE].min_k[gaggle_id] = min_k;
	// Print bits min_k
	
}
// bit procssing time
void  set_bit(unsigned char A[],  int k )
{
    A[k/8] |= 1 << (k%8);  // Set the bit at the k-th position in A[i]
}

void  clear_bit(unsigned char A[],  int k )                
{
    A[k/8] &= ~(1 << (k%8));
}

int test_bit(unsigned char A[],  int k )
{
    return ( (A[k/8] & (1 << (k%8) )) != 0 ) ;     
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned char value, unsigned int intial_position, unsigned int size)
{   

    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
            clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned long value, unsigned int intial_position, unsigned int size)
{
    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
           clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned short value, unsigned int intial_position, unsigned int size)
{
    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
           clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

void swap_char(unsigned char *data_array, unsigned char value, unsigned int array_pos)
{   
    
     for (unsigned int i = 0; i < 8; ++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((8 -1) - i) + array_pos * 8);
        }
        else
        {
            clear_bit(data_array, ((8 -1) - i) + array_pos * 8);
        }
    }

}

void read_array(FILE *ofp, unsigned char *data_array, unsigned int size, unsigned int array_size)
{

    for (unsigned int i = 0; i < array_size; ++i)
    {   
        swap_char(data_array,data_array[i], i);
        fprintf(ofp, "%02X ",data_array[i]);
    }
}


void convert_to_bits_segment(FILE *ofp, block_struct segment_data)
{
	std::string data_out = "";
	unsigned int total_bits = 0;
	// get the total bits needed
	// bits in header 
	total_bits += 160;
	unsigned int gaggle_index = 0;
	unsigned int gaggle_id = 0;
	unsigned int gaggles = 0;
	// bits in shifted_dc
	if (segment_data.N == 1)
	{
		total_bits += segment_data.header->part3.seg_size_blocks_20bits; // shifted DC is size 1 bit for numer of blocks so number of blocks
	}
	else
	{
		// DC encoder
		while(gaggle_index < segment_data.header->part3.seg_size_blocks_20bits)
		{
			gaggles = GAGGLE_SIZE > (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) ? (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
			
			// min K
			total_bits += segment_data.bits_min_k; // add one per gaggle in the block list
			unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
			// mappedDC
				for (unsigned int i = gaggle_index; i < gaggle_index + gaggles; ++i)
				{
					if ((segment_data.min_k[gaggle_id] == uncoded_flag) || (i==0))
					{
						total_bits += segment_data.N;
					}
					else
					{
						total_bits +=(segment_data.dc_mapper[i] >> segment_data.min_k[gaggle_id]) +1;
					}
				}
				// if we have coded samples, then we also have to send the second part
				if (segment_data.min_k[gaggle_id] != uncoded_flag)
				{	
					int max_value = gaggle_index > 1 ? gaggle_index : 1; // max operation (づ｡◕‿‿◕｡)づ
					for(unsigned int i = max_value; i < gaggle_index + gaggles; ++i)
					{
						// print_out dc_mapper[i], min_k
						total_bits += segment_data.min_k[gaggle_id];
					}
				}
			gaggle_index += gaggles;
			gaggle_id += 1;
		}
	}
	// bits DC renaning
	total_bits += segment_data.numaddbitplanes * segment_data.header->part3.seg_size_blocks_20bits;
	// total bits calculate
	unsigned int array_size = ceil(float(total_bits)/float(8));
	//unsigned int last_position_number_bits = total_bits % 8;
	// declarate data array
	unsigned char *data_array = NULL;
	data_array = (unsigned char*)calloc(array_size, sizeof(unsigned char));
	
	// print header
	// part 1
	unsigned int position = 0;
	position = write_data_to_array(data_array, segment_data.header->part1.start_img_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.end_img_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.segment_count_8bits ,position,8);
	position = write_data_to_array(data_array, segment_data.header->part1.bit_depth_dc_5bits ,position,5);
	position = write_data_to_array(data_array, segment_data.header->part1.bit_depth_ac_5bits ,position,5);
	position = write_data_to_array(data_array, segment_data.header->part1.reserved ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_2_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_3_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_4_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.pad_rows_3bits ,position,3);
	position = write_data_to_array(data_array, segment_data.header->part1.reserved_5bits ,position,5);

	// part 2
	position = write_data_to_array(data_array, segment_data.header->part2.seg_byte_limit_27bits ,position,27);
	position = write_data_to_array(data_array, segment_data.header->part2.dc_stop ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.bit_plane_stop_5bits ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.stage_stop_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part2.use_fill ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.reserved_4bits ,position,4);

	// part 3
	position = write_data_to_array(data_array, segment_data.header->part3.seg_size_blocks_20bits ,position,20);
	position = write_data_to_array(data_array, segment_data.header->part3.opt_dc_select ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part3.opt_ac_select ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part3.reserved_2bits ,position,2);

	// part 3
	position = write_data_to_array(data_array, segment_data.header->part4.dwt_type ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.reserved_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.signed_pixels ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.pixel_bit_depth_4bits ,position,4);
	position = write_data_to_array(data_array, segment_data.header->part4.image_with_20bits ,position,20);
	position = write_data_to_array(data_array, segment_data.header->part4.transpose_img ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.code_word_length ,position,3);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LL3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.reserved_11bits ,position,11);
	// print shifted_dc
	if (segment_data.N == 1)
	{
		for (unsigned int i = 0; i < segment_data.header->part3.seg_size_blocks_20bits; ++i )
		{
			position = write_data_to_array(data_array, segment_data.shifted_dc_1bit[i] ,position,1);
		}
	}
	else
	{
		// pint DC encoder
		gaggle_index = 0;
		gaggle_id = 0;
		gaggles = 0;
		while(gaggle_index < segment_data.header->part3.seg_size_blocks_20bits)
		{
			gaggles = GAGGLE_SIZE > (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) ? (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
			
			// min K
			total_bits += segment_data.bits_min_k; // add one per gaggle in the block list
			unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
			// mappedDC
				for (unsigned int i = gaggle_index; i < gaggle_index + gaggles; ++i)
				{
					if ((segment_data.min_k[gaggle_id] == uncoded_flag) || (i==0))
					{
						position = write_data_to_array(data_array, segment_data.dc_mapper[i] ,position,segment_data.N);
					}
					else
					{
						position = write_data_to_array(data_array, (unsigned char)1 ,position,(segment_data.dc_mapper[i] >> segment_data.min_k[gaggle_id]) +1);
						
					}
				}
				// if we have coded samples, then we also have to send the second part
				if (segment_data.min_k[gaggle_id] != uncoded_flag)
				{	
					int max_value = gaggle_index > 1 ? gaggle_index : 1; // max operation (づ｡◕‿‿◕｡)づ
					for(unsigned int i = max_value; i < gaggle_index + gaggles; ++i)
					{
						// print_out dc_mapper[i], min_k
						position = write_data_to_array(data_array, segment_data.dc_mapper[i] ,position,segment_data.min_k[gaggle_id]);
					}
				}
			gaggle_index += gaggles;
			gaggle_id += 1;
		}
	}
	for (unsigned int i = 0; i < segment_data.numaddbitplanes; ++i)
	{
		for(unsigned int k = 0; k < segment_data.header->part3.seg_size_blocks_20bits; ++k)
		{
			position = write_data_to_array(data_array, (unsigned char)(segment_data.dc_remainer[k] >> (segment_data.quantization_factor - i -1) ),position,1);
		}
	}


	// print the array
	read_array(ofp, data_array,total_bits, array_size);



}

unsigned long conversion_twos_complement(long original, short leftmost)
{

	unsigned long temp;
	unsigned long complement;

	short i = 0;
	if (leftmost == 1)
	{
		return 0;
	}
	
	if (leftmost >= sizeof(unsigned long) * 8 || (leftmost == 0))
	{
		return 0; // ERROR IN THE DATA
	}

	if (original >=0)
	{
		return (unsigned long) original;
	}
	else
	{
		complement = ~(unsigned long) (-original);
		temp = 0;
		for (i = 0; i < leftmost; ++i)
		{
			temp <<= 1;
			++temp;
		}
		complement &= temp;
		complement ++;
		return complement;
	}
	
}

void clean(DataObject *device_object){
    cudaFree(device_object->input_image);

    for(unsigned int x = 0; x < NUMBER_STREAMS ; ++x){
        cudaStreamDestroy(cuda_streams[x]);
    }

    delete device_object->start_bpe;
    delete device_object->stop_bpe;
    delete device_object->start_dwt;
    delete device_object->stop_dwt;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
    delete device_object;
}
