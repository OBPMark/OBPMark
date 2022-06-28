
/**
 * \file procesing.h
 * \brief Benchmark #2.1 CUDA processing header.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#ifndef PROCESSING_CUDA_H_
#define PROCESSING_CUDA_H_


/** 
 * \brief This 
 */
__global__ void
process_input_preprocessor(
    const unsigned int * input_data, 
    unsigned  int *input_data_post_process, 
    int* zero_block_list, 
    int* zero_block_list_inverse, 
    int block_size, 
    int number_blocks,
    unsigned int n_bits);


/** 
 * \brief This.
 */
__global__ void
process_input_no_preprocessor(
    const unsigned  int * input_data, 
    unsigned  int *input_data_post_process, 
    int* zero_block_list, 
    int* zero_block_list_inverse, 
    int block_size, 
    int number_blocks);

/** 
 * \brief This.
 */

__global__ void
zero_block_list_completition(
    int* zero_block_list, 
    int* zero_block_list_inverse, 
    int *missing_value, 
    int *missing_value_inverse, 
    int stream, 
    int block_size, 
    int number_blocks);


/**
 * @brief This.
 */
__global__ void
 adaptative_entropy_encoder_no_compresion(
    unsigned int *input_data_post_process, 
    int *zero_block_list, 
    unsigned  int *data_in_blocks ,
    unsigned int *size_block ,
    unsigned char *compresion_identifier,
    unsigned char * compresion_identifier_internal,
    unsigned int id ,
    int block_size, 
    int number_blocks, 
    int number_bits);


/**
 * @brief This.
 */
__global__ void
adaptative_entropy_encoder_zero_block(
    unsigned int *input_data_post_process, 
    int *zero_block_list, 
    int *zero_block_list_inverse, 
    unsigned  int *data_in_blocks ,
    unsigned int *size_block ,
    unsigned char *compresion_identifier,
    unsigned char *compresion_identifier_internal,
    unsigned int id ,
    int block_size, 
    int number_blocks, 
    int number_bits);

/**
 * @brief This.
 */
__global__ void
adaptative_entropy_encoder_second_extension(
    unsigned int *input_data_post_process, 
    int *zero_block_list, 
    unsigned int *data_in_blocks ,
    unsigned int *size_block ,
    unsigned int *halved_samples,
    unsigned char *compresion_identifier, 
    unsigned char *compresion_identifier_internal,
    unsigned int id ,
    int block_size, 
    int number_blocks, 
    int number_bits);


/**
 * @brief This.
 */
 __global__ void
adaptative_entropy_encoder_sample_spliting(
    unsigned  int *input_data_post_process, 
    int *zero_block_list, 
    unsigned  int *data_in_blocks ,
    unsigned int *size_block ,
    unsigned char *compresion_identifier, 
    unsigned char *compresion_identifier_internal,
    unsigned int id ,
    int block_size, 
    int number_blocks, 
    int number_bits);

/**
 * @brief This.
 */
__global__ void
adaptative_entropy_encoder_block_selector(
    int *zero_block_list ,
    unsigned int *bit_block_best,
    unsigned int *size_block ,
    unsigned char *compresion_identifier, 
    unsigned char *compresion_identifier_internal ,
    unsigned int *size_block_best ,
    unsigned char *compresion_identifier_best, 
    unsigned char *compresion_identifier_internal_best,
    int block_size, 
    int number_blocks, 
    int number_bits);

/**
 * @brief This.
 */
__global__ void
adaptative_entropy_encoder_block_selector_data_copy(
    int *zero_block_list, 
    unsigned  int *data_in_blocks ,
    unsigned int *bit_block_best, 
    unsigned  int *data_in_blocks_best ,
    int block_size, 
    int number_blocks);



#endif // PROCESSING_CUDA_H_