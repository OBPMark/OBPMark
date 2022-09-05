/** 
 * \brief OBPMark "Image compression algorithm." processing task and image kernels.
 * \file processing.h
 * \author Ivan Rodriguez-Ferrandez (BSC)
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "obpmark.h"
#include "device.h"



/**
 * \brief Computes the three levels of DWT to get the image in the correct format for the compression. 
 */
void dwt2D_compression_computation(
	compression_image_data_t *compression_data,
    int  **image_data,
    int  **transformed_image,
    unsigned int h_size_padded,
	unsigned int w_size_padded
	);
/**
 * \brief regroups the DWT levels to get the image in the correct format for the decompression. 
 */

void build_block_string(
    int **transformed_image, 
    unsigned int h_size, 
    unsigned int w_size, 
    int **block_string
    );



/**
 * \brief takes block_string and computes each of the blocks to compute the DC and AC coefficients. 
 */
void compute_bpe(
    compression_image_data_t *compression_data,
    int **block_string,
    unsigned int num_segments
    );


#endif // PROCESSING_H_