#htvar kernel_code
#define x_min 0

void kernel process_input_no_preprocessor(global const unsigned long int * input_data, unsigned int shift_input_data, global unsigned long int *input_data_post_process,unsigned int shift_input_data_post,global int* zero_block_list,global int* zero_block_list_inverse, unsigned int shift_zero_block, int block_size, int number_blocks)
{
    // iter over the numer of blocks
    const unsigned int i = get_global_id(0);
    // shiftin
    input_data = input_data + shift_input_data;
    input_data_post_process = input_data_post_process + shift_input_data_post;
    zero_block_list = zero_block_list + shift_zero_block;
    zero_block_list_inverse = zero_block_list_inverse + shift_zero_block;

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

void kernel process_input_preprocessor(global const unsigned long int * input_data, unsigned int shift_input_data, global unsigned long int *input_data_post_process,unsigned int shift_input_data_post,global int* zero_block_list,global int* zero_block_list_inverse, unsigned int shift_zero_block, int block_size, int number_blocks)
{
    // iter over the numer of blocks
    const unsigned int i = get_global_id(0);
    // shiftin
    input_data = input_data + shift_input_data;
    input_data_post_process = input_data_post_process + shift_input_data_post;
    zero_block_list = zero_block_list + shift_zero_block;
    zero_block_list_inverse = zero_block_list_inverse + shift_zero_block;

    if ( i < number_blocks)
    {
        // first process all data in the blocks this could be maximun to 64 so will perform this opertaion in GPU
        int total_value = 0;
        for ( unsigned int x = 0; x < block_size; ++ x)
        {
            // unit delay input
            unsigned long int pre_value = i == 0 && x== 0 ? 0 : input_data[x + (i * block_size) -1];
            
            
            const int prediction_error = input_data[x + (i * block_size)] - pre_value;
            const int theta = min((unsigned long)(pre_value - x_min), (unsigned long)(pow( (float)2,(float)(n_bits - 1)) - pre_value));
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

void kernel zero_block_list_completition(global int* zero_block_list, global int* zero_block_list_inverse, unsigned int shift_zero_block, global int *missing_value, global int *missing_value_inverse, int stream, int block_size, int number_blocks)
{
    const unsigned int i = get_global_id(0);
    zero_block_list = zero_block_list + shift_zero_block;
    zero_block_list_inverse = zero_block_list_inverse + shift_zero_block;

    if ( i * 2 < number_blocks)
    {
        // first step
        if(i != 0)
        {
            if(zero_block_list[i*2] == 1 && zero_block_list[i*2 + 1] == 1)
            {
                zero_block_list[i*2] = -1;
                zero_block_list[i*2 + 1] = -1;
                atomic_add(&missing_value[stream],2);
            }
            else if(zero_block_list[i*2] == 1)
            {
                zero_block_list[i*2] = -1;
                atomic_add(&missing_value[stream],1);

            }
            // inverse part
            if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1 && zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2)] = -1;
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomic_add(&missing_value_inverse[stream],2);
            }
            else if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomic_add(&missing_value_inverse[stream],1);

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
        
        barrier(CLK_LOCAL_MEM_FENCE);
        int step = 0;
        while(missing_value[stream] != 0) // TODO:FIXMI JAQUER only works in predetermine case 
        {
            if(i != 0)
            {   
                if(zero_block_list[(i*2) - (step % 2)] != -1 && zero_block_list[(i*2) + 1 - (step % 2)] == -1)
                {
                    zero_block_list[(i*2) + 1 - (step % 2)] = zero_block_list[(i*2) - (step % 2)] + 1;
                    atomic_add(&missing_value[stream], -1);
                }
                // inverse part
                if(zero_block_list_inverse[(number_blocks -1) - ((i*2) - (step % 2))] != -1 && zero_block_list_inverse[(number_blocks -1) - ((i*2) + 1 - (step % 2))] == -1)
                {
                    zero_block_list_inverse[(number_blocks -1) - ((i*2) + 1 - (step % 2))] = zero_block_list_inverse[(number_blocks -1) - ((i*2) - (step % 2))] + 1;
                    atomic_add(&missing_value_inverse[stream], -1);
                }
                                
            }
            step += 1;
            barrier(CLK_LOCAL_MEM_FENCE);
        }


    }

}


void kernel adaptative_entropy_encoder_no_compresion(global unsigned long int *input_data_post_process, unsigned int shift_data_post_process,global int *zero_block_list,global int *zero_block_list_inverse,unsigned int shift_zero_block, global unsigned long int *data_in_blocks, unsigned int shift_data_in_blocks ,global unsigned int *size_block ,global unsigned char *compresion_identifier, unsigned int shift_size_compresion,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = get_global_id(0);
    input_data_post_process = input_data_post_process + shift_data_post_process;
    zero_block_list = zero_block_list + shift_zero_block;
    zero_block_list_inverse = zero_block_list_inverse + shift_zero_block;
    data_in_blocks = data_in_blocks + shift_data_in_blocks;
    size_block = size_block +shift_size_compresion;
    compresion_identifier = compresion_identifier + shift_size_compresion;


    if ( i < number_blocks)
    {
        if(zero_block_list[i] == 0)
        {
            // no zero block
            // memory position
            size_block[i + (id * number_blocks)] = number_bits * block_size;
            compresion_identifier[i + (id * number_blocks)] = 32; // NO COMPRESION ID
            // acces each data in the array of data
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            input_data_post_process[(i * block_size)] = 2;
            for (unsigned int x = 0; x < block_size ;++x)
            {
                #pragma unroll
                for(unsigned int bit = 0; bit < number_bits; ++bit)
                {   
                    if((input_data_post_process[((x * 32 + bit)/32) + (i * block_size)] & (1 << ((x*32 + bit)%32) )) != 0)
                    {
                        
                        data_in_blocks[((x*n_bits+bit)/32) + base_position_data] |= 1 << ((x*number_bits+bit)%32);
                    }
                }
            }
        }
        else
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
void kernel adaptative_entropy_encoder_second_extension(global unsigned long int *input_data_post_process, unsigned int shift_data_post_process,global int *zero_block_list,unsigned int shift_zero_block, global unsigned long int *data_in_blocks , unsigned int shift_data_in_blocks,global unsigned int *size_block ,global unsigned char *compresion_identifier,unsigned int shift_size_compresion, unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = get_global_id(0);
    input_data_post_process = input_data_post_process + shift_data_post_process;
    zero_block_list = zero_block_list + shift_zero_block;
    data_in_blocks = data_in_blocks + shift_data_in_blocks;
    size_block = size_block +shift_size_compresion;
    compresion_identifier = shift_size_compresion + compresion_identifier;


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

void kernel adaptative_entropy_encoder_sample_spliting(global unsigned long int *input_data_post_process, unsigned int shift_data_post_process,global int *zero_block_list,unsigned int shift_zero_block, global unsigned long int *data_in_blocks , unsigned int shift_data_in_blocks,global unsigned int *size_block ,global unsigned char *compresion_identifier,unsigned int shift_size_compresion,unsigned int id ,int block_size, int number_blocks, int number_bits)
{
    const unsigned int i = get_global_id(0);
    input_data_post_process = input_data_post_process + shift_data_post_process;
    zero_block_list = zero_block_list + shift_zero_block;
    data_in_blocks = data_in_blocks + shift_data_in_blocks;
    size_block = size_block +shift_size_compresion;
    compresion_identifier = shift_size_compresion + compresion_identifier;
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


void kernel adaptative_entropy_encoder_block_selector(global int *zero_block_list,unsigned int shift_zero_block, global unsigned long int *data_in_blocks, unsigned int shift_data_in_blocks ,global unsigned int *size_block ,global unsigned char *compresion_identifier ,unsigned int shift_size_compresion,global unsigned long int *data_in_blocks_best ,unsigned int shift_block_best, global unsigned int *size_block_best ,global unsigned char *compresion_identifier_best,unsigned int shift_size_compresion_best,int block_size, int number_blocks, int number_bits)
{
    // select the best only one can survive 
    const unsigned int i = get_global_id(0);
    zero_block_list = zero_block_list + shift_zero_block;
    data_in_blocks = data_in_blocks + shift_data_in_blocks;
    size_block = size_block + shift_size_compresion;
    compresion_identifier = compresion_identifier + shift_size_compresion;
    data_in_blocks_best = data_in_blocks_best + shift_block_best;
    size_block_best = size_block_best + shift_size_compresion_best;
    compresion_identifier_best =  compresion_identifier_best + shift_size_compresion_best;
    
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
            const int base_position_data = (best_id * block_size * number_blocks) + (i * block_size);
            size_block_best[i] = size_block[i + (best_id * number_blocks)];
            compresion_identifier_best[i] = compresion_identifier[i + (best_id * number_blocks)];
            // copy data
            for(unsigned int x = 0; x < block_size; ++x)
            {
                data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
            }

        }
        else
        {
            // is 0 block
            const int base_position_data = (best_id * block_size * number_blocks) + (i * block_size);
            size_block_best[i] = size_block[i + (best_id * number_blocks)];
            compresion_identifier_best[i] = compresion_identifier[i + (best_id * number_blocks)];
            // copy data
            for(unsigned int x = 0; x < block_size; ++x)
            {
                data_in_blocks_best[(i* block_size) + x] = data_in_blocks[base_position_data + x];
            }
        }
        
        
        
    }
}
void kernel post_processing_of_output_data(global unsigned long int *data_in_blocks_best,unsigned int shift_block_best ,global unsigned int *size_block_best,global unsigned char *compresion_identifier_best,unsigned int shift_size_compresion_best,global unsigned long int *data_in_blocks_best_post_process,unsigned int shift_post_process,int block_size, int number_blocks)
{
    const unsigned int i = get_global_id(0); // number of blocks
    const unsigned int j = get_global_id(1); // 32 * block size

    data_in_blocks_best = data_in_blocks_best + shift_block_best;
    size_block_best = size_block_best +shift_size_compresion_best;
    compresion_identifier_best = shift_size_compresion_best + compresion_identifier_best;
    
    data_in_blocks_best_post_process = data_in_blocks_best_post_process + shift_post_process;


    if (i < number_blocks && j < 32 * block_size)
    {
        if (j == 0) // ones per block size * 32 
        {
            const unsigned short size = size_block_best[i];
            data_in_blocks_best_post_process[i * block_size] = compresion_identifier_best[i] + (size << 6);
            size_block_best[i] +=  22;
        }
        
        // Bit shifting compressed array 22 bits to input the header (size and identification method).
        if((data_in_blocks_best[j/32 + (i * block_size)] & (1 << j%32)) != 0)
        {
            data_in_blocks_best_post_process[(((j+22)/32  + (i * block_size)))] |= 1 << ((j+22)%32);
        } 
    }


}
						                             																		                                                                      																										                                    
#htendvar