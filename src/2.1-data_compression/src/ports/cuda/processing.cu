
/**
* \file processing.c
* \brief Benchmark #2.1 CUDA kernel implementation.
* \author Ivan Rodriguez-Ferrandez (BSC)
*/
#include "device.h"
#include "processing.h"
#include "obpmark.h"
#include "obpmark_time.h"

///////////////////////////////////////////////////////////////////////////////////////////////
// KERNELS
///////////////////////////////////////////////////////////////////////////////////////////////


__global__ void
process_input_preprocessor(const unsigned int * input_data, unsigned int *input_data_post_process, int* zero_block_list_status, int* zero_block_list, int* zero_block_list_inverse, int block_size, int number_blocks, unsigned int n_bits)
{
    // iter over the numer of blocks
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x_min =  0;
    const unsigned int x_max = pow( 2,n_bits) - 1;
    if ( i < number_blocks)
    {
        // first process all data in the blocks this could be maximun to 64 so will perform this opertaion in GPU
        int total_value = 0;
        for ( unsigned int x = 0; x < block_size; ++ x)
        {
            // unit delay input
            unsigned int pre_value = i == 0 && x== 0 ? 0 : input_data[x + (i * block_size) -1];
            
            
            const int prediction_error = input_data[x + (i * block_size)] - pre_value;
            const int theta = min((unsigned long)(pre_value - x_min), (unsigned long)(x_max - pre_value));
            int preprocess_sample = theta  + abs(prediction_error);
            
            // predictor error mapper
            input_data_post_process[x + (i * block_size)] =  0 <= prediction_error && prediction_error <= theta ? 2 * prediction_error :  (-theta <= prediction_error && prediction_error < 0 ? ((2 * abs(prediction_error)) -1): preprocess_sample);

            // Zero block detection
            total_value += input_data_post_process[x + (i * block_size)];

        }
        // update the zero_block_data
        zero_block_list_status[i] =   total_value > 0 ? 0 : 1;
        zero_block_list[i] =  total_value > 0 ? 0 : 1;
        zero_block_list_inverse[i] = total_value > 0 ? 0 : 1;
       
        
    }
    
}



__global__ void
process_input_no_preprocessor(const unsigned int * input_data, unsigned int *input_data_post_process,int* zero_block_list_status, int* zero_block_list, int* zero_block_list_inverse, int block_size, int number_blocks)
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
        zero_block_list_status[i] =   total_value > 0 ? 0 : 1;
        zero_block_list[i] =   total_value > 0 ? 0 : 1;
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
            //if (stream == 0){printf("%d\n",missing_value[stream]);}
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
adaptative_entropy_encoder_no_compresion(unsigned int *input_data_post_process, int *zero_block_list, unsigned  int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier,unsigned char * compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    if ( i < number_blocks && x < block_size)
    {
        if(zero_block_list[i] == 0)
        {
            // no zero block
            // memory position
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            // if is first element in the block do the size and the compresion identifier
            if(x == 0){
                size_block[i + (id * number_blocks)] = number_bits * block_size;
                compresion_identifier_internal[i + (id * number_blocks)] = 32; // NO COMPRESION ID

                // create a ternary tree for selection the compression identifier
                compresion_identifier[i + (id * number_blocks)] = number_bits < 3 ? 0x1 : number_bits < 5 ? 0x3 : number_bits <= 8 ? 0x7 : number_bits <= 16 ? 0xF : 0x1F;
                
            }
            // copy the data
            //printf("i:%d x:%d: %d\n",i,x,input_data_post_process[base_position_data + x]);
            data_in_blocks[base_position_data + x] = input_data_post_process[base_position_data + x];
           
        }
    }

}


__global__ void
adaptative_entropy_encoder_zero_block(unsigned int *input_data_post_process, int *zero_block_list, int *zero_block_list_inverse, unsigned int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier, unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        if(zero_block_list[i] != 0)
        {
            // compute ROS
            /*if (zero_block_list[i] < 5)
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
            }*/
            size_block[i + (id * number_blocks)] = zero_block_list[i];
            compresion_identifier[i + (id * block_size)] = ZERO_BLOCK_ID; // NO COMPRESION ID
            compresion_identifier_internal[i + (id * block_size)] = ZERO_BLOCK_ID;
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            data_in_blocks[(base_position_data)] = 0;
            
        }
    }

}

__global__ void
adaptative_entropy_encoder_second_extension(unsigned int *input_data_post_process, int *zero_block_list, unsigned int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier, unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int halved_samples_internal[MAX_NUMBER_OF_BLOCKS];
    if ( i < number_blocks)
    {
        
        if(zero_block_list[i] == 0)
        {
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            // no zero block so do the second extension
            //identifier
            compresion_identifier[i + (id * number_blocks)] = SECOND_EXTENSION_ID -1; // Second Extension
            compresion_identifier_internal[i + (id * number_blocks)] = SECOND_EXTENSION_ID; // Second Extension
            size_block[i + (id * number_blocks)] = block_size * 32;
            // clear the data
            // calculate thing
            for(unsigned int x = 0; x < block_size/2;++x)
            {
                halved_samples_internal[x] = (((input_data_post_process[((2*x) + (i * block_size))] + input_data_post_process[((2*x) + (i * block_size)) + 1]) * (input_data_post_process[((2*x) + (i * block_size))] + input_data_post_process[((2*x) + (i * block_size)) + 1] + 1)) / 2) + input_data_post_process[((2*x) + (i * block_size)) + 1];
            }
            // get size
            unsigned int size = 0;
            
            // get size
            for(int x = 0; x <  block_size/2; ++x)
            {
                size += halved_samples_internal[x] + 1;
                // store output
            }
            // store size
            if(size < (block_size * 32))
            {
                size_block[i + (id * number_blocks)] = size;
                for(int x = 0; x <  block_size/2; ++x)
                {
                    // store output
                    data_in_blocks[base_position_data + x] = halved_samples_internal[x];
                }
            }

        }
    }
}



__global__ void
adaptative_entropy_encoder_sample_spliting(unsigned  int *input_data_post_process, int *zero_block_list, unsigned  int *data_in_blocks ,unsigned int *size_block ,unsigned char *compresion_identifier, unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < number_blocks)
    {
        if(zero_block_list[i] == 0)
        {

            // no zero block so do the sample spliting
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            const unsigned int k = (id - 2);
            compresion_identifier[i + (id * number_blocks)] = SAMPLE_SPLITTING_ID + k  - 2; // SAMPLE_SPLITTING_ID
            compresion_identifier_internal[i + (id * number_blocks)] = SAMPLE_SPLITTING_ID + k ; // SAMPLE_SPLITTING_ID
            if(k >= number_bits -2)
            {
                size_block[i + (id * number_blocks)] = block_size * 32;
            }
            else if (k == 0)
            {  
                // get fundamental sequence
                compresion_identifier[i + (id * number_blocks)] = 1; // FUNDAMENTAL_SEQUENCE_ID
                compresion_identifier_internal[i + (id * number_blocks)] = 1; // FUNDAMENTAL_SEQUENCE_ID
                // first get size and process the output at the same time
                unsigned int size = 0;
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
                        data_in_blocks[base_position_data + x]  = input_data_post_process[(x + (i * block_size))];
                    }
                }
                

            }
            else
            {
                // sample spliting when k != 0 and k >= n_bits -3
                unsigned int size = 0;
                for(int x = 0; x < block_size; ++x)
                {   
                    // get the size
                    size += (k + (input_data_post_process[(x + (i * block_size))] >> k)) + 1;
                    data_in_blocks[base_position_data + x]  = input_data_post_process[(x + (i * block_size))];
                }

                size_block[i + (id * number_blocks)] = size;
            }
        }
    }

}


__global__ void
adaptative_entropy_encoder_block_selector(int *zero_block_list ,unsigned int *bit_block_best,unsigned int *size_block ,unsigned char *compresion_identifier, unsigned char *compresion_identifier_internal ,unsigned int *size_block_best ,unsigned char *compresion_identifier_best, unsigned char *compresion_identifier_internal_best,int block_size, int number_blocks, int number_bits)
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
            for(unsigned int method = 0; method < number_bits + 2; ++ method)
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
            compresion_identifier_internal_best[i] = compresion_identifier_internal[i + (best_id * number_blocks)];

        }
        else
        {
            // is 0 block
            size_block_best[i] = size_block[i + (best_id * number_blocks)];
            compresion_identifier_best[i] =   ZERO_BLOCK_ID; 
            compresion_identifier_internal_best[i] = ZERO_BLOCK_ID;
        }
        
        
        
    }
}



__global__ void
adaptative_entropy_encoder_block_selector_data_copy(int *zero_block_list, unsigned  int *data_in_blocks ,unsigned int *bit_block_best, unsigned  int *data_in_blocks_best ,int block_size, int number_blocks)
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
            //const int base_position_data = (0 * block_size * number_blocks) + (i * block_size);
            data_in_blocks_best[(i* block_size)] = 1;
        }
    }
    


}