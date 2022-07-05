#htvar kernel_code


void kernel
process_input_preprocessor(global unsigned int * input_data, global unsigned int *input_data_post_process,global int* zero_block_list,global int* zero_block_list_inverse, int block_size, int number_blocks, unsigned int n_bits, unsigned int x_max, unsigned int offset_input_data_internal, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list)
{
    // iter over the numer of blocks
    const unsigned int i = get_global_id(0);
    const unsigned int x_min =  0;
    // update pointers
    input_data += offset_input_data_internal;
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    zero_block_list_inverse += offset_zero_block_list;
    // iterate over the block
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
            total_value += input_data[x + (i * block_size)];

        }
        // update the zero_block_data
        
        zero_block_list[i] =  total_value > 0 ? 0 : 1;
        zero_block_list_inverse[i] = total_value > 0 ? 0 : 1;
       
        
    }
    
}



void kernel
process_input_no_preprocessor(global const unsigned int * input_data,global unsigned int *input_data_post_process,global int* zero_block_list,global int* zero_block_list_inverse, int block_size, int number_blocks, unsigned int offset_input_data_internal, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list)
{
    // iter over the numer of blocks
    const unsigned int i = get_global_id(0);
    // update pointers
    input_data += offset_input_data_internal;
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    zero_block_list_inverse += offset_zero_block_list;
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


void kernel
zero_block_list_completition(global int* zero_block_list,global int* zero_block_list_inverse,global int *missing_value,global int *missing_value_inverse, int stream, int block_size, int number_blocks, unsigned int offset_zero_block_list)
{
    const unsigned int i = get_global_id(0);
    // update pointers
    zero_block_list += offset_zero_block_list;
    zero_block_list_inverse += offset_zero_block_list;
    if ( i * 2 < number_blocks)
    {
        // first step
        if(i != 0)
        {
            if(zero_block_list[i*2] == 1 && zero_block_list[i*2 + 1] == 1)
            {
                zero_block_list[i*2] = -1;
                zero_block_list[i*2 + 1] = -1;
                atomic_add (&missing_value[stream],2);
            }
            else if(zero_block_list[i*2] == 1)
            {
                zero_block_list[i*2] = -1;
                atomic_add (&missing_value[stream],1);

            }
            // inverse part
            if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1 && zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2)] = -1;
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomic_add (&missing_value_inverse[stream],2);
            }
            else if(zero_block_list_inverse[(number_blocks -1) - (i*2)] == 1)
            {
                zero_block_list_inverse[(number_blocks -1) - (i*2 + 1)] = -1;
                atomic_add (&missing_value_inverse[stream],1);

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



void kernel
adaptative_entropy_encoder_no_compresion(global unsigned int *input_data_post_process,global int *zero_block_list,global unsigned  int *data_in_blocks ,global unsigned int *size_block ,global unsigned char *compresion_identifier,global unsigned char * compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list, unsigned int offset_data_in_blocks, unsigned int offset_size_block, unsigned int offset_compresion_identifier)
{
    const unsigned int i = get_global_id(0);
    const unsigned int x = get_global_id(1);
    // update pointers
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    data_in_blocks += offset_data_in_blocks;
    size_block += offset_size_block;
    compresion_identifier += offset_compresion_identifier;
    compresion_identifier_internal += offset_compresion_identifier;

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
            /*for (unsigned int x = 0; x < block_size ;++x)
            {
               
                data_in_blocks[base_position_data + x] = input_data_post_process[x + (i * block_size)];
            }*/
        }
        
    }

}


void kernel
adaptative_entropy_encoder_zero_block(global unsigned int *input_data_post_process,global int *zero_block_list,global int *zero_block_list_inverse,global unsigned int *data_in_blocks ,global unsigned int *size_block ,global unsigned char *compresion_identifier,global unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list, unsigned int offset_zero_block_list_inverse, unsigned int offset_data_in_blocks, unsigned int offset_size_block, unsigned int offset_compresion_identifier)
{
    const unsigned int i = get_global_id(0);
    // update pointers
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    zero_block_list_inverse += offset_zero_block_list_inverse;
    data_in_blocks += offset_data_in_blocks;
    size_block += offset_size_block;
    compresion_identifier += offset_compresion_identifier;
    compresion_identifier_internal += offset_compresion_identifier;
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
            compresion_identifier[i + (id * block_size)] = ZERO_BLOCK_ID; // NO COMPRESION ID
            compresion_identifier_internal[i + (id * block_size)] = ZERO_BLOCK_ID;
            const int base_position_data = (id * block_size * number_blocks) + (i * block_size);
            data_in_blocks[(base_position_data)] = 1;
            
        }
    }

}

void kernel
adaptative_entropy_encoder_second_extension(global unsigned int *input_data_post_process,global int *zero_block_list,global unsigned int *data_in_blocks ,global unsigned int *size_block ,global unsigned int *halved_samples,global unsigned char *compresion_identifier,global unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list, unsigned int offset_data_in_blocks, unsigned int offset_size_block, unsigned int offset_halved_samples, unsigned int offset_compresion_identifier)
{
    const unsigned int i = get_global_id(0);
    // update pointers
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    data_in_blocks += offset_data_in_blocks;
    size_block += offset_size_block;
    halved_samples += offset_halved_samples;
    compresion_identifier += offset_compresion_identifier;
    compresion_identifier_internal += offset_compresion_identifier;
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
            if(size < (block_size * 32))
            {
                
                size_block[i + (id * number_blocks)] = size;
                for(int x = 0; x <  block_size/2; ++x)
                {
                    // store output
                    data_in_blocks[base_position_data + x] = halved_samples[x];
                }
            }

        }
    }
}



void kernel
adaptative_entropy_encoder_sample_spliting(global unsigned  int *input_data_post_process,global int *zero_block_list,global unsigned  int *data_in_blocks ,global unsigned int *size_block ,global unsigned char *compresion_identifier,global unsigned char *compresion_identifier_internal,unsigned int id ,int block_size, int number_blocks, int number_bits, unsigned int offset_input_data_post_process, unsigned int offset_zero_block_list, unsigned int offset_data_in_blocks, unsigned int offset_size_block, unsigned int offset_compresion_identifier)
{
    const unsigned int i = get_global_id(0);
    // update pointers
    input_data_post_process += offset_input_data_post_process;
    zero_block_list += offset_zero_block_list;
    data_in_blocks += offset_data_in_blocks;
    size_block += offset_size_block;
    compresion_identifier += offset_compresion_identifier;
    compresion_identifier_internal += offset_compresion_identifier;

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


void kernel
adaptative_entropy_encoder_block_selector(global int *zero_block_list ,global unsigned int *bit_block_best,global unsigned int *size_block ,global unsigned char *compresion_identifier,global unsigned char *compresion_identifier_internal ,global unsigned int *size_block_best ,global unsigned char *compresion_identifier_best,global unsigned char *compresion_identifier_internal_best,int block_size, int number_blocks, int number_bits, unsigned int offset_zero_block_list, unsigned int offset_bit_block_best, unsigned int offset_size_block, unsigned int offset_compresion_identifier , unsigned int offset_size_block_best, unsigned int offset_compresion_identifier_best)
{
    // select the best only one can survive 
    const unsigned int i = get_global_id(0);
    // update pointers
    zero_block_list += offset_zero_block_list;
    bit_block_best += offset_bit_block_best;
    size_block += offset_size_block;
    compresion_identifier += offset_compresion_identifier;
    compresion_identifier_internal += offset_compresion_identifier;
    size_block_best += offset_size_block_best;
    compresion_identifier_best += offset_compresion_identifier_best;
    compresion_identifier_internal_best += offset_compresion_identifier_best;
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
            /*if (best_id == 0)
            {
                 compresion_identifier_best[i] = number_bits < 3 ? 0x1 : number_bits < 5 ? 0x3 : number_bits <= 8 ? 0x7 : number_bits <= 16 ? 0xF : 0x1F;
                 compresion_identifier_internal_best[i] = 32;
            }
            else if (best_id == 1)
            {
                compresion_identifier_best[i] = SECOND_EXTENSION_ID -1;
                compresion_identifier_internal_best[i] = SECOND_EXTENSION_ID;
            }
            else if (best_id == 2)
            {
                compresion_identifier_best[i] = 1;
                compresion_identifier_internal_best[i] = 1;
            }
            else
            {
                compresion_identifier_best[i] = SAMPLE_SPLITTING_ID + best_id  - 4;
                compresion_identifier_internal_best[i] = SAMPLE_SPLITTING_ID + best_id  - 4;
            }*/
            compresion_identifier_best[i] = compresion_identifier[i + (best_id * number_blocks)];
            compresion_identifier_internal_best[i] = compresion_identifier_internal[i + (best_id * number_blocks)];
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



void kernel
adaptative_entropy_encoder_block_selector_data_copy(global int *zero_block_list,global unsigned  int *data_in_blocks ,global unsigned int *bit_block_best,global unsigned  int *data_in_blocks_best ,int block_size, int number_blocks, unsigned int offset_zero_block_list, unsigned int offset_data_in_blocks, unsigned int offset_bit_block_best, unsigned int offset_data_in_blocks_best)
{
    const unsigned int i = get_global_id(0);
    const unsigned int x = get_global_id(1);

    // update pointers
    zero_block_list += offset_zero_block_list;
    data_in_blocks += offset_data_in_blocks;
    bit_block_best += offset_bit_block_best;
    data_in_blocks_best += offset_data_in_blocks_best;

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
#htendvar