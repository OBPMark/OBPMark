#include "lib_functions.h"
#include "Config.h"
#include "AdaptativeEntropyEncoder.h"
#include "Preprocessor.h"

#define x_min 0
#define x_max pow( 2,n_bits) - 1


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d code id: %d\n", cudaGetErrorString(code), file, line, code);
      if (abort) exit(code);
   }
}

//###############################################################################
//# Kernels
//###############################################################################
__global__ void
process_input_no_preprocessor(const unsigned long int * input_data, unsigned long int *input_data_post_process, int* zero_block_list, int* zero_block_list_inverse, int block_size, int number_blocks)
{
    // iter over the numer of blocks
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        // first process all data in the blocks this could be maximun to 64 so will perform this opertaion in GPU
        int total_value = 0;
        for ( unsigned int x = 0; x < block_size; ++ x)
        {
            // only copy
            input_data_post_process[x + (i * block_size)] = input_data[x + (i * block_size)];
            total_value += input_data[x + (i * block_size)];

        }
        // update the zero_block_data
        zero_block_list[i] =   total_value > 0 ? 0 : 1;
        zero_block_list_inverse[i] = total_value > 0 ? 0 : 1;
    }
  
    
}

__global__ void
process_input_preprocessor(const unsigned long int * input_data, unsigned long int *input_data_post_process, int* zero_block_list, int* zero_block_list_inverse, int block_size, int number_blocks)
{
    // iter over the numer of blocks
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        // first process all data in the blocks this could be maximun to 64 so will perform this opertaion in GPU
        int total_value = 0;
        for ( unsigned int x = 0; x < block_size; ++ x)
        {
            // unit delay input
            unsigned long int pre_value = i == 0 && x== 0 ? 0 : input_data[x + (i * block_size) -1];
            
            
            const int prediction_error = input_data[x + (i * block_size)] - pre_value;
            const int theta = min((unsigned long)(pre_value - x_min), (unsigned long)(x_max - pre_value));
            int preprocess_sample = theta  + abs(prediction_error);
            
            // predictor error mapper
            input_data_post_process[x + (i * block_size)] =  0 <= prediction_error && prediction_error <= theta ? 2 * prediction_error :  (-theta <= prediction_error && prediction_error < 0 ? ((2 * abs(prediction_error)) -1): preprocess_sample);

            // Zero block detection
            total_value += input_data[x + (i * block_size)];

        }
        // update the zero_block_data
        
        zero_block_list[i] =  total_value > 0 ? 0 : 1;
        zero_block_list_inverse[i] = total_value > 0 ? 0 : 1;
       
        
    }
    
}

__global__ void
zero_block_list_completition(int* zero_block_list, int* zero_block_list_inverse, int *missing_value, int *missing_value_inverse, int stream, int block_size, int number_blocks)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i * 2 < number_blocks)
    {
        // first step
        if(i != 0)
        {
            if(zero_block_list[i*2] == 1 && zero_block_list[i*2 + 1] == 1)
            {
                zero_block_list[i*2] = -1;
                zero_block_list[i*2 + 1] = -1;
                atomicAdd(&missing_value[stream],2);
            }
            else if(zero_block_list[i*2] == 1)
            {
                zero_block_list[i*2] = -1;
                atomicAdd(&missing_value[stream],1);

            }
            // inverse part
            if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1 && zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2)] = -1;
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomicAdd(&missing_value_inverse[stream],2);
            }
            else if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomicAdd(&missing_value_inverse[stream],1);

            }
        }
        else
        {
            if(zero_block_list[0] == 1 && zero_block_list[1] == 1)
            {
                zero_block_list[1] = 2;
               
            }
            // inverse part
            if(zero_block_list_inverse[(number_blocks -1)] == 1 && zero_block_list_inverse[(number_blocks -1) - 1] == 1)
            {
                zero_block_list_inverse[(number_blocks -1)-1] = 2;
               
            }
             
             
        }
        
        __syncthreads();
        int step = 0;
        while(missing_value[stream] != 0) // TODO:FIXMI JAQUER only works in predetermine case 
        {
            if(i != 0)
            {   
                if(zero_block_list[(i*2) - (step % 2)] != -1 && zero_block_list[(i*2) + 1 - (step % 2)] == -1)
                {
                    zero_block_list[(i*2) + 1 - (step % 2)] = zero_block_list[(i*2) - (step % 2)] + 1;
                    atomicAdd(&missing_value[stream], -1);
                }
                // inverse part
                if(zero_block_list_inverse[(number_blocks -1) - ((i*2) - (step % 2))] != -1 && zero_block_list_inverse[(number_blocks -1) - ((i*2) + 1 - (step % 2))] == -1)
                {
                    zero_block_list_inverse[(number_blocks -1) - ((i*2) + 1 - (step % 2))] = zero_block_list_inverse[(number_blocks -1) - ((i*2) - (step % 2))] + 1;
                    atomicAdd(&missing_value_inverse[stream], -1);
                }
                                
            }
            step += 1;
            __syncthreads();
        }


    }

}

__global__ void
adaptative_entropy_encoder_zero_block(unsigned long int *input_data_post_process, int *zero_block_list, int *zero_block_list_inverse, unsigned long int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        if(zero_block_list[i] != 0)
        {
            // compute ROS
            if (zero_block_list[i] < 5)
            {
                size_block[i + (id * number_blocks)] = zero_block_list[i];
                
            }
            else if(zero_block_list[i] == 5 && zero_block_list_inverse[i] >= 5)
            {
                // ROS
                size_block[i + (id * number_blocks)] = zero_block_list[i];
            }
            else if (zero_block_list[i] >= 5 && zero_block_list_inverse[i] >= 5)
            {
                size_block[i + (id * number_blocks)] = zero_block_list[i] + 1;
            } 
            else
            {
                size_block[i + (id * number_blocks)] = zero_block_list[i] + 1;
            }
            compresion_identifier[i + (id * block_size)] = 0; // NO COMPRESION ID
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            data_in_blocks[(base_position_data)] = 1;
            
        }
    }

}

__global__ void
adaptative_entropy_encoder_no_compresion(unsigned long int *input_data_post_process, int *zero_block_list, int *zero_block_list_inverse, unsigned long int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int bit = blockIdx.z * blockDim.z + threadIdx.z;
    if ( i < number_blocks && x < block_size && bit < number_bits)
    {
        if(zero_block_list[i] == 0)
        {
            // no zero block
            // memory position
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            if(x == 0 && bit == 0){
                size_block[i + (id * number_blocks)] = number_bits * block_size;
                compresion_identifier[i + (id * number_blocks)] = 32; // NO COMPRESION ID
                // acces each data in the array of data
                input_data_post_process[(i * block_size)] = 2;
            }
            //for (unsigned int x = 0; x < block_size ;++x)
            //{
                //#pragma unroll
                //for(unsigned int bit = 0; bit < number_bits; ++bit)
                //{   
                    if((input_data_post_process[((x * 32 + bit)/32) + (i * block_size)] & (1 << ((x*32 + bit)%32) )) != 0)
                    {
                        
                        data_in_blocks[((x*n_bits+bit)/32) + base_position_data] |= 1 << ((x*number_bits+bit)%32);
                    }
                //}
            //}
        }
    }

}

__global__ void
adaptative_entropy_encoder_second_extension(unsigned long int *input_data_post_process, int *zero_block_list, unsigned long int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int halved_samples[J_BlockSize/2];
    if ( i < number_blocks)
    {
        
        if(zero_block_list[i] == 0)
        {
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            // no zero block so do the second extension
            //identifier
            compresion_identifier[i + (id * number_blocks)] = 1; // Second Extension
            size_block[i + (id * number_blocks)] = block_size * 32;
            // calculate thing
            for(unsigned int x = 0; x < block_size/2;++x)
            {
                halved_samples[x] = (( (input_data_post_process[((2*x) + (i * block_size))] + input_data_post_process[((2*x) + (i * block_size)) + 1]) * (input_data_post_process[((2*x) + (i * block_size))] + input_data_post_process[((2*x) + (i * block_size)) + 1] + 1)) / 2) + input_data_post_process[((2*x) + (i * block_size)) + 1];
            }
            // get size
            unsigned int size = 0;
            
            // get size
            for(int x = 0; x <  block_size/2; ++x)
            {
                size += halved_samples[x] + 1;
                // store output
            }
            // store size
            
            unsigned int sample = 0;
            if(size < (block_size * 32))
            {
                
                size_block[i + (id * number_blocks)] = size;
                for(int x = 0; x <  block_size/2; ++x)
                {
                    // store output
                    data_in_blocks[base_position_data + (sample/32)] |= 1 << (sample%32);
                    sample += halved_samples[x] + 1;
                }
            }

        }
    }
}

__global__ void
adaptative_entropy_encoder_sample_spliting(unsigned long int *input_data_post_process, int *zero_block_list, unsigned long int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        if(zero_block_list[i] == 0)
        {

            // no zero block so do the sample spliting
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            const unsigned int k = (id - 2);
            compresion_identifier[i + (id * number_blocks)] = 2 + k; // SAMPLE_SPLITTING_ID
            if(k >= n_bits -3)
            {
                size_block[i + (id * number_blocks)] = block_size * 32;
            }
            else if (k == 0)
            {  
                // get fundamental sequence
                // first get size and process the output at the same time
                unsigned int size = 0;
                unsigned int sample = 0; 
                for(int x = 0; x < block_size; ++x)
                {
                    // get the size
                    size += input_data_post_process[(x + (i * block_size))] + 1;
                }
                size_block[i + (id * number_blocks)] = size;
                if(size < block_size * 32)
                {
                    for(int x = 0; x < block_size; ++x)
                    {
                        // create output
                        data_in_blocks[base_position_data + (sample/32)]  |= 1 << (sample%32);
                        sample += input_data_post_process[(x + (i * block_size))] + 1;
                    }
                }
                

            }
            else
            {
                // sample spliting when k != 0 and k >= n_bits -3
                unsigned int fssample = 0; 
                unsigned int least_significantve_bits = 0;
                unsigned int offset = 0;
                unsigned int size = 0;
                for(int x = 0; x < block_size; ++x)
                {
                    // get the size
                    size += input_data_post_process[(x + (i * block_size))] + 1;
                    fssample = input_data_post_process[(x + (i * block_size))] >> k;
                    least_significantve_bits = input_data_post_process[(x + (i * block_size))] & ((1UL << k) - 1);
                    data_in_blocks[base_position_data + (offset/32)] |= least_significantve_bits << (offset%32);
                    data_in_blocks[base_position_data + ((offset + k)/32)] |= 1 << ((offset+k)%32);
                    offset += (k + fssample + 1);
                }
                size_block[i + (id * number_blocks)] = size;
            }
        }
    }

}


__global__ void
adaptative_entropy_encoder_block_selector(int *zero_block_list ,unsigned int *bit_block_best,unsigned int *size_block ,unsigned char *compresion_identifier ,unsigned int *size_block_best ,unsigned char *compresion_identifier_best,int block_size, int number_blocks, int number_bits)
{
    // select the best only one can survive 
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        unsigned int best_id = 0;
        if(zero_block_list[i] == 0)
        {
            // is not zero block
            unsigned int size_minmun = 999999;
            for(unsigned int method = 0; method < n_bits + 2; ++ method)
            {
                if (size_block[i + (method * number_blocks)] < size_minmun)
                {
                    size_minmun = size_block[i + (method * number_blocks)];
                    best_id = method;
                }
            }
            
            // best ID geted Now process
            //const int base_position_data = (best_id * block_size * number_blocks) + (i * block_size);
            bit_block_best[i] = best_id;
            size_block_best[i] = size_block[i + (best_id * number_blocks)];
            compresion_identifier_best[i] = compresion_identifier[i + (best_id * number_blocks)];
            // copy data
            /*for(unsigned int x = 0; x < block_size; ++x)
            {
                data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
            }*/

        }
        else
        {
            // is 0 block
            //const int base_position_data = (best_id * block_size * number_blocks) + (i * block_size);
            size_block_best[i] = size_block[i + (best_id * number_blocks)];
            compresion_identifier_best[i] = compresion_identifier[i + (best_id * number_blocks)];
            // copy data
            //for(unsigned int x = 0; x < block_size; ++x)
            //{
            //    data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
            //}
        }
        
        
        
    }
}

__global__ void
adaptative_entropy_encoder_block_selector_data_copy(int *zero_block_list, unsigned long int *data_in_blocks ,unsigned int *bit_block_best, unsigned long int *data_in_blocks_best ,int block_size, int number_blocks)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i < number_blocks && x < block_size)
    {
        if(zero_block_list[i] == 0)
        {
            const int base_position_data = (bit_block_best[i] * block_size * number_blocks) + (i * block_size);
            data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
        }
        else
        {
            const int base_position_data = (0 * block_size * number_blocks) + (i * block_size);
            data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
        }
    }
    


}
__global__ void
post_processing_of_output_data(unsigned long int *data_in_blocks_best  ,unsigned char *compresion_identifier_best, unsigned int *size_block_best,unsigned long int *data_in_blocks_best_post_process, int block_size, int number_blocks)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // number of blocks
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; // 32 * block size

    if (i < number_blocks && j < 32 * block_size)
    {
        if (j == 0) // ones per block size * 32 
        {
            const unsigned short size = size_block_best[i];
            data_in_blocks_best_post_process[i * block_size] = compresion_identifier_best[i] + (size << 6);
            size_block_best[i] +=  22;
        }
        
        // Bit shifting compressed array 22 bits to input the header (size and identification method).
        //if(i==0 && j==0){printf("%lu \n\n", data_in_blocks_best_post_process[i * block_size]);}
        if((data_in_blocks_best[j/32 + (i * block_size)] & (1 << j%32)) != 0)
        {
            data_in_blocks_best_post_process[(((j+22)/32  + (i * block_size)))] |= 1 << ((j+22)%32);
        } 
        //if(i==0 && j==0){printf("%lu \n\n", data_in_blocks_best_post_process[i * block_size]);}
    }


}
//###############################################################################
cudaStream_t cuda_streams[NUMBER_STREAMS][2 + n_bits];
//void copy_data_to_cpu(DataObject *device_object);
void copy_data_to_cpu_asycronous(DataObject *device_object, int step);

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
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}

bool device_memory_init(struct DataObject *device_object)
{   
    // Allocate the device image imput
    cudaError_t err = cudaSuccess;
    // data input

    err = cudaMalloc((void **)&device_object->input_data, sizeof( unsigned long int ) * device_object->TotalSamplesStep);
    if (err != cudaSuccess)
    {
        return false;
    }
    // data post_processed
    err = cudaMalloc((void **)&device_object->input_data_post_process, sizeof( unsigned long int ) * device_object->TotalSamples * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }
    // data out

    err = cudaMalloc((void **)&device_object->output_data, sizeof( unsigned long int ) * device_object->TotalSamplesStep);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->missing_value, sizeof(int) * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->missing_value_inverse, sizeof(int) * NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->zero_block_list, sizeof(int) * r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->zero_block_list_inverse, sizeof(int) * r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }
    // Aee variables 
    err = cudaMalloc((void **)&device_object->size_block, sizeof(unsigned int) * r_samplesInterval *  NUMBER_STREAMS * (2 + n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->compresion_identifier, sizeof(unsigned char) * r_samplesInterval *  NUMBER_STREAMS * (2 + n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->data_in_blocks_best, sizeof(unsigned long int)  * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS );
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->data_in_blocks_best_post_process, sizeof(unsigned long int)  * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS );
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->size_block_best, sizeof(unsigned int) * r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->bit_block_best, sizeof(unsigned int) * r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->compresion_identifier_best, sizeof(unsigned char) * r_samplesInterval *  NUMBER_STREAMS);
    if (err != cudaSuccess)
    {
        return false;
    }

    err = cudaMalloc((void **)&device_object->data_in_blocks, sizeof(unsigned long int) * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS * (2 + n_bits));
    if (err != cudaSuccess)
    {
        return false;
    }

    for(unsigned int x = 0; x < NUMBER_STREAMS; ++x){
        for(unsigned int y = 0; y < 2 + n_bits; ++ y){
            cudaStreamCreate(&cuda_streams[x][y]);
        }
    }
    // int CPU part
    device_object->data_in_blocks_best_cpu = ( unsigned long int *)malloc(sizeof( unsigned long int ) * device_object->TotalSamplesStep);
    device_object->size_block_best_cpu = ( unsigned int *)malloc(sizeof(unsigned int) * r_samplesInterval * STEPS);
    device_object->compresion_identifier_best_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * r_samplesInterval * STEPS);
    return true;
}
void copy_data_to_gpu(DataObject *device_object)
{
    cudaEventRecord(*device_object->start_memory_copy_device);
    cudaError_t err = cudaMemcpy(device_object->input_data, device_object->InputDataBlock, sizeof( unsigned long int ) * device_object->TotalSamplesStep, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input_image from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
	cudaEventRecord(*device_object->stop_memory_copy_device);
}

void execute_benchmark(struct DataObject *device_object)
{

    copy_data_to_gpu(device_object);
    //gpuErrchk( cudaPeekAtLastError() );
    //cudaEventRecord(*device_object->start_app);
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_app);

    // Repeating the operations n times
    for(int step = 0; step < STEPS; ++step)
    {
        #ifdef PREPROCESSOR_ACTIVE

        // Preprocesor active
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(r_samplesInterval)/dimBlock.x));
        process_input_preprocessor<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->input_data + (device_object->TotalSamples * step),
        device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)) ,
        device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS)),
        device_object->zero_block_list_inverse + (r_samplesInterval * (step % NUMBER_STREAMS)),
        J_BlockSize,
        r_samplesInterval);
        //gpuErrchk( cudaPeekAtLastError() );
        #else

        // Not preprocessor
        dim3 dimBlock(BLOCK_SIZE*BLOCK_SIZE);
        dim3 dimGrid(ceil(float(r_samplesInterval)/dimBlock.x));
        process_input_no_preprocessor<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->input_data + (device_object->TotalSamples * step),
        device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)) ,
        device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS)),
        device_object->zero_block_list_inverse + (r_samplesInterval * (step % NUMBER_STREAMS)),
        J_BlockSize,
        r_samplesInterval);
        //gpuErrchk( cudaPeekAtLastError() );
        
        #endif
        // copy data and detect the zero blockls
        dimBlock = dim3(BLOCK_SIZE*BLOCK_SIZE);
        dimGrid = dim3(ceil(float(r_samplesInterval/2)/dimBlock.x));
        // indentif the total reminaing zero blocks
        zero_block_list_completition<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->zero_block_list+ (r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->zero_block_list_inverse+ (r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->missing_value,
            device_object->missing_value_inverse,
            (step % NUMBER_STREAMS),
            J_BlockSize,
            r_samplesInterval);
        //gpuErrchk( cudaPeekAtLastError() );
        // zero block finish 
        // start  adaptative entropy encoder
        // sync stream
        cudaStreamSynchronize (cuda_streams[step % NUMBER_STREAMS][0]);
        //gpuErrchk( cudaStreamSynchronize (cuda_streams[step % NUMBER_STREAMS][0]));
        //gpuErrchk( cudaPeekAtLastError() );
        // processing of each block for aee
        // launch no_compresion and 0 block
        dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,n_bits);
        dimGrid = dim3(ceil(float(r_samplesInterval)/dimBlock.x), ceil(float(J_BlockSize)/dimBlock.y), 1);
        adaptative_entropy_encoder_no_compresion<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)), 
            device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS) ) ,
            device_object->zero_block_list_inverse + (r_samplesInterval * (step % NUMBER_STREAMS) ) , 
            device_object->data_in_blocks + (r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->size_block + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->compresion_identifier + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            0,
            J_BlockSize,
            r_samplesInterval,
            n_bits);
        //gpuErrchk( cudaPeekAtLastError() );
        dimBlock = dim3(BLOCK_SIZE*BLOCK_SIZE);
        dimGrid = dim3(ceil(float(r_samplesInterval)/dimBlock.x));
        adaptative_entropy_encoder_zero_block<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)), 
            device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS) ) ,
            device_object->zero_block_list_inverse + (r_samplesInterval * (step % NUMBER_STREAMS) ) , 
            device_object->data_in_blocks + (r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->size_block + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->compresion_identifier + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            0,
            J_BlockSize,
            r_samplesInterval,
            n_bits);
        //gpuErrchk( cudaPeekAtLastError() );
        // launch second extension
       adaptative_entropy_encoder_second_extension<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][1]>>>(device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)), 
            device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS) ) , 
            device_object->data_in_blocks + (r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->size_block + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->compresion_identifier + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            1,
            J_BlockSize,
            r_samplesInterval,
            n_bits);
        //gpuErrchk( cudaPeekAtLastError() );

        // launch sample spiting
        for (unsigned int bit = 0; bit < n_bits; ++ bit)
        {
            adaptative_entropy_encoder_sample_spliting<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][bit + 2]>>>(device_object->input_data_post_process + (device_object->TotalSamples * (step % NUMBER_STREAMS)), 
                device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS) ) , 
                device_object->data_in_blocks + (r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits)),
                device_object->size_block + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
                device_object->compresion_identifier + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
                bit + 2,
                J_BlockSize,
                r_samplesInterval,
                n_bits);
            //gpuErrchk( cudaPeekAtLastError() );
        }
        // sync screams of the same "primary" stream
        //gpuErrchk( cudaPeekAtLastError() );
        for(unsigned int y = 0; y < 2 + n_bits; ++ y)
        {
            cudaStreamSynchronize(cuda_streams[step % NUMBER_STREAMS][y]);
        }
        //gpuErrchk( cudaPeekAtLastError() );
        // block selector 
        adaptative_entropy_encoder_block_selector<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS)) , 
            device_object->bit_block_best + (J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->size_block + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->compresion_identifier + (r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)),
            device_object->size_block_best + (r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->compresion_identifier_best + (r_samplesInterval * (step % NUMBER_STREAMS)),
            J_BlockSize,
            r_samplesInterval,
            n_bits);

        dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
        dimGrid = dim3(ceil(float(r_samplesInterval)/dimBlock.x),ceil(float(J_BlockSize)/dimBlock.y));
        adaptative_entropy_encoder_block_selector_data_copy<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->zero_block_list + (r_samplesInterval * (step % NUMBER_STREAMS)) ,
            device_object->data_in_blocks + (r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits)),   
            device_object->bit_block_best + (J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->data_in_blocks_best + (J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS)),
            J_BlockSize,
            r_samplesInterval);
        // finish acclerated gpu
        // sync stream
        //gpuErrchk( cudaPeekAtLastError() );
        //cudaStreamSynchronize (cuda_streams[step % NUMBER_STREAMS][0]);
        //gpuErrchk( cudaPeekAtLastError() );
        // precompute the data
        //#########################################################################
        
        //#########################################################################
        dimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
        dimGrid = dim3(ceil(float(r_samplesInterval)/dimBlock.x), ceil(float(32 * J_BlockSize)/dimBlock.y));
        post_processing_of_output_data<<<dimGrid,dimBlock,0, cuda_streams[step % NUMBER_STREAMS][0]>>>(device_object->data_in_blocks_best + (J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->compresion_identifier_best + (r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->size_block_best + (r_samplesInterval * (step % NUMBER_STREAMS)),
            device_object->data_in_blocks_best_post_process + (J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS)),
            J_BlockSize,
            r_samplesInterval);
        //gpuErrchk( cudaPeekAtLastError() );
        // copy data back
        copy_data_to_cpu_asycronous(device_object, step);
        //gpuErrchk( cudaPeekAtLastError() );
        
    }
    // sync GPU
    cudaDeviceSynchronize();
    //gpuErrchk( cudaPeekAtLastError() );
    //printf("\n\n\n\n");
    /*for(unsigned int x = 0; x < r_samplesInterval; ++x)
    {
        printf("%u %u %lu|", device_object->size_block_best_cpu[x], device_object->compresion_identifier_best_cpu[x], device_object->data_in_blocks_best_cpu[x * J_BlockSize]);
    }
    for(unsigned int x = 0; x < r_samplesInterval; ++x)
    {
        printf("Size: %u, id: %u, data: %lu\n", device_object->size_block_best_cpu[x], (unsigned int)device_object->compresion_identifier_best_cpu[x],device_object->data_in_blocks_best_cpu[x*J_BlockSize]);
    }*/
    // create output
    unsigned int acc_size = 0;
    for(int step = 0; step < STEPS; ++step)
    {    
        // header
        for(unsigned char bit = 0; bit < 6; ++bit)
        {
            if((n_bits & (1 << (bit%8) )) != 0)
            {
                device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
            }
        }
        acc_size += 6;
        for(unsigned int block = 0; block < r_samplesInterval; ++block)
        {  

            // reprocess the data
            
            for(int bit = 0; bit < device_object->size_block_best_cpu[block + (step * r_samplesInterval)]; ++bit)
            {
                
                if((device_object->data_in_blocks_best_cpu[(bit/32) + (block * J_BlockSize) + (step * r_samplesInterval * J_BlockSize)] & (1 << (bit%32) )) != 0)
                {
                    device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
                }
            }
            acc_size += device_object->size_block_best_cpu[block + (step * r_samplesInterval)];
            
        
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_app);
    //cudaEventRecord(*device_object->stop_app);
    // Debug loop - uncomment if needed
    //printf("%d\n\n", acc_size);
    for(int i = acc_size - 1; i >= 0; --i)
    {
        printf("%d" ,(device_object->OutputDataBlock[i/32] & (1 << (i%32) )) != 0);
    }
    printf("\n");
    
    
    

}
void copy_data_to_cpu_asycronous(DataObject *device_object, int step)
{
    cudaEventRecord(*device_object->start_memory_copy_host);

    cudaMemcpyAsync(device_object->data_in_blocks_best_cpu + (J_BlockSize * r_samplesInterval * step), device_object->data_in_blocks_best_post_process + ((step % NUMBER_STREAMS) * J_BlockSize * r_samplesInterval), sizeof( unsigned long int ) * r_samplesInterval * J_BlockSize, cudaMemcpyDeviceToHost, cuda_streams[step % NUMBER_STREAMS][0]);
    cudaMemcpyAsync(device_object->size_block_best_cpu + (r_samplesInterval * step), device_object->size_block_best + ((step % NUMBER_STREAMS)  * r_samplesInterval), sizeof( unsigned int ) * r_samplesInterval , cudaMemcpyDeviceToHost,cuda_streams[step % NUMBER_STREAMS][1]);
    cudaMemcpyAsync(device_object->compresion_identifier_best_cpu + (r_samplesInterval * step), device_object->compresion_identifier_best + ((step % NUMBER_STREAMS)  * r_samplesInterval), sizeof( unsigned char ) * r_samplesInterval , cudaMemcpyDeviceToHost,cuda_streams[step % NUMBER_STREAMS][2]);
    
    cudaEventRecord(*device_object->stop_memory_copy_host);

}

/*void copy_data_to_cpu(DataObject *device_object)
{   
	
    cudaEventRecord(*device_object->start_memory_copy_host);

    cudaMemcpy(device_object->data_in_blocks_best_cpu, device_object->data_in_blocks_best, sizeof( unsigned long int ) * device_object->TotalSamplesStep, cudaMemcpyDeviceToHost);
    cudaMemcpy(device_object->size_block_best_cpu, device_object->size_block_best, sizeof( unsigned int ) * r_samplesInterval , cudaMemcpyDeviceToHost);
    cudaMemcpy(device_object->compresion_identifier_best_cpu, device_object->compresion_identifier_best, sizeof( unsigned char ) * r_samplesInterval , cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
   
}*/

void get_elapsed_time(DataObject *device_object, bool csv_format){
    //cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time 1
    long unsigned int application_miliseconds = (device_object->end_app.tv_sec - device_object->start_app.tv_sec) * 1000 + (device_object->end_app.tv_nsec - device_object->start_app.tv_nsec) / 1000000;
    //cudaEventElapsedTime(&milliseconds, *device_object->start_app, *device_object->stop_app);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    // part 2 of the BPE
    //miliseconds_bpe += (device_object->end_bpe_cpu.tv_sec - device_object->start_bpe_cpu.tv_sec) * 1000 + (device_object->end_bpe_cpu.tv_nsec - device_object->start_bpe_cpu.tv_nsec) / 1000000;

    if (csv_format){
         printf("%.10f;%lu;%.10f;\n", milliseconds_h_d,application_miliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %lu miliseconds\n", application_miliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
}


void clean(struct DataObject *device_object)
{
    free(device_object->InputDataBlock);
    free(device_object->OutputDataBlock);
    free(device_object);
}