/** 
 * \brief BPMark "Image compression algorithm." processing task and image kernels.
 * \file processing.c
 * \author Ivan Rodriguez-Ferrandez (BSC)
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "processing.h"


void coefficient_scaling (int **transformed_image,unsigned int h_size_padded,unsigned int w_size_padded);
void coeff_regroup(int **transformed_image,unsigned int h_size_padded,unsigned int w_size_padded);


/**
 * \brief Section that computes the DWT 1D 
 */

void ccsds_wavelet_transform_1D(const int* A, int* B, const int size){
	// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
	unsigned int full_size = size * 2;
	// integer computation
	// high part
	for (unsigned int i = 0; i < size; ++i){
		int sum_value_high = 0;
		// specific cases
		if(i == 0){
			sum_value_high = A[1] - (int)( ((9.0/16.0) * (A[0] + A[2])) - ((1.0/16.0) * (A[2] + A[4])) + (1.0/2.0));
		}
		else if(i == size -2){
			sum_value_high = A[2*size - 3] - (int)( ((9.0/16.0) * (A[2*size -4] + A[2*size -2])) - ((1.0/16.0) * (A[2*size - 6] + A[2*size - 2])) + (1.0/2.0));
		}
		else if(i == size - 1){
			sum_value_high = A[2*size - 1] - (int)( ((9.0/8.0) * (A[2*size -2])) -  ((1.0/8.0) * (A[2*size - 4])) + (1.0/2.0));
		}
		else{
			// generic case
			sum_value_high = A[2*i+1] - (int)( ((9.0/16.0) * (A[2*i] + A[2*i+2])) - ((1.0/16.0) * (A[2*i - 2] + A[2*i + 4])) + (1.0/2.0));
		}
		
		//store
		B[i+size] = sum_value_high;

	

	}
	// low_part
	for (unsigned int i = 0; i < size; ++i){
		int sum_value_low = 0;
		if(i == 0){
			sum_value_low = A[0] - (int)(- (B[size]/2.0) + (1.0/2.0));
		}
		else
		{
			sum_value_low = A[2*i] - (int)( - (( B[i + size -1] +  B[i + size])/ 4.0) + (1.0/2.0) );
		}
		
		B[i] = sum_value_low;
	}
}
	
void ccsds_wavelet_transform_1D(const float* A, float* B, const int size){
	// flotating part
	unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
	int gi_end = HIGHPASSFILTERSIZE / 2;

	for (unsigned int i = 0; i < size ; i = ++i ){
		// loop over N elements of the input vector.
		float sum_value_low = 0;
		// first process the lowpass filter
		for (int hi = hi_start; hi < hi_end + 1; ++hi){
			int x_position = (2 * i) + hi;
			if (x_position < 0) {
				// turn negative to positive
				x_position = x_position * -1;
			}
			else if (x_position > full_size - 1)
			{
				x_position = full_size - 1 - (x_position - (full_size -1 ));
			}
			// now I need to restore the hi value to work with the array
			sum_value_low += lowpass_filter_cpu[hi + hi_end] * A[x_position];
			
		}
		// store the value
		B[i] = sum_value_low;
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
			sum_value_high += highpass_filter_cpu[gi + gi_end] * A[x_position];
		}
		// store the value
		B[i+size] = sum_value_high;
	}
	
}


/**
 * \brief Section that computes the DWT 2D 
 */
void ccsds_wavelet_transform_2D (int** A, const int w_size, const int h_size){
    // auxiliar pointer to the intermidiate part
	int *auxiliar = (int *)malloc(sizeof(int) * h_size * w_size);
    // we nee to be perform a 1D DWT in all rows and another 1D DWT in the columns side
	for (unsigned int i = 0; i < h_size; ++i){   
        ccsds_wavelet_transform_1D(A[i], auxiliar + i * w_size, w_size/2);
    }
	int *auxiliar_h_input = (int *)malloc(sizeof(int) * h_size);
	int *auxiliar_h_output = (int *)malloc(sizeof(int) * h_size);
    for (unsigned int i = 0; i < w_size; ++i){
		// for each iteration create an auxiliar array
		for(unsigned int x = 0; x < h_size; ++x ){
			auxiliar_h_input[x] = auxiliar[x *w_size + i];
		}
        ccsds_wavelet_transform_1D(auxiliar_h_input, auxiliar_h_output, h_size/2);
		for(unsigned int x = 0; x < h_size; ++x ){
			A[x][i] = auxiliar_h_output[x];
		}
    }
	free(auxiliar);
	free(auxiliar_h_input);
	free(auxiliar_h_output);
}

void ccsds_wavelet_transform_2D (float** A, const int w_size, const int h_size)
{
	// auxiliar pointer to the intermidiate part
	float *auxiliar = (float *)malloc(sizeof(float) * h_size * w_size);
    // we nee to be perform a 1D DWT in all rows and another 1D DWT in the columns side
	for (unsigned int i = 0; i < h_size; ++i){   
        ccsds_wavelet_transform_1D(A[i], auxiliar + i * w_size, w_size/2);
    }
	float *auxiliar_h_input = (float *)malloc(sizeof(float) * h_size);
	float *auxiliar_h_output = (float *)malloc(sizeof(float) * h_size);
    for (unsigned int i = 0; i < w_size; ++i){
		// for each iteration create an auxiliar array
		for(unsigned int x = 0; x < h_size; ++x ){
			auxiliar_h_input[x] = auxiliar[x *w_size + i];
		}
        ccsds_wavelet_transform_1D(auxiliar_h_input, auxiliar_h_output, h_size/2);
		for(unsigned int x = 0; x < h_size; ++x ){
			A[x][i] = auxiliar_h_output[x];
		}
    }
	free(auxiliar);
	free(auxiliar_h_input);
	free(auxiliar_h_output);
}


/**
 * \brief Section that computes the DWT 2D three levels with the two types of data computations
 */
void dwt2D_compression_computation_integer(
    compression_image_data_t *compression_data,
    int  **image_data,
    int  **transformed_image
    )
{
    unsigned int h_size_padded = compression_data->h_size + compression_data->pad_rows;
	unsigned int w_size_padded = compression_data->w_size + compression_data->pad_columns;
	unsigned int pad_rows = compression_data->pad_rows;
	unsigned int pad_colums = compression_data->pad_columns;
    unsigned int iteration = 0;
    while(iteration != LEVELS_DWT){
        if (iteration == 0){
            ccsds_wavelet_transform_2D(image_data,w_size_padded, h_size_padded);
        }
        else{
            // create a subimage from the original
            unsigned int new_h_size = compression_data->h_size/ (2*iteration);
            unsigned int new_w_size = compression_data->w_size/ (2*iteration);
            int  **aux_image = NULL;
            aux_image = (int**)calloc(new_h_size, sizeof(int *));
            for(unsigned i = 0; i < new_h_size; i++){
                aux_image[i] = (int *)calloc(w_size_padded, sizeof(int));
            }
            // copy the subimage
            for( int i=0 ; i < new_h_size ; i++){
                for( int j=0 ; j < new_w_size ; j++){
                    aux_image[i][j] = image_data[i][j];
                }
            }
            // send subimage
            ccsds_wavelet_transform_2D(aux_image, new_w_size, new_h_size);
            // copy back image
            for( int i=0 ; i < new_h_size ; i++){
                for( int j=0 ; j < new_w_size ; j++){
                    image_data[i][j] = aux_image[i][j];
                }
            }
            for(unsigned i = 0; i < new_h_size; i++){
                free(aux_image[i]);
            }
            free(aux_image);
        }	
        
        ++iteration;
    }
    // copy the image
    for(unsigned int i = 0; i < h_size_padded; i++)
    {		
        for(unsigned int j = 0; j < w_size_padded; j++)		
        {
                transformed_image[i][j] = image_data[i][j];
        }
    }
}

void dwt2D_compression_computation_float(
    compression_image_data_t *compression_data,
    int  **image_data,
    int  **transformed_image
    )
{   
    unsigned int h_size_padded = compression_data->h_size + compression_data->pad_rows;
    unsigned int w_size_padded = compression_data->w_size + compression_data->pad_columns;
    unsigned int pad_rows = compression_data->pad_rows;
    unsigned int pad_colums = compression_data->pad_columns;

    float  **aux_data = NULL;
    aux_data = (float**)calloc(h_size_padded, sizeof(float *));
    for(unsigned i = 0; i < h_size_padded; i++){
        aux_data[i] = (float *)calloc(w_size_padded, sizeof(float));
    }
    // convert the original interger image unsigned int to float
    // convert image to float
    for( int i=0 ; i < compression_data->h_size ; i++)
    {
        for( int j=0 ; j < compression_data->w_size ; j++)
        {
            aux_data[i][j] =  float(image_data[i][j]);
        }
    }
    unsigned int iteration = 0;
    while(iteration != LEVELS_DWT){
        if (iteration == 0){
            ccsds_wavelet_transform_2D(aux_data,w_size_padded, h_size_padded);
        }
        else{
            // create a subimage from the original
            unsigned int new_h_size = h_size_padded / (2*iteration);
            unsigned int new_w_size = w_size_padded / (2*iteration);
            float  **aux_image = NULL;
            aux_image = (float**)calloc(new_h_size, sizeof(float *));
            for(unsigned i = 0; i < new_h_size; i++){
                aux_image[i] = (float *)calloc(w_size_padded, sizeof(float));
            }
            // copy the subimage
            for( int i=0 ; i < new_h_size ; i++){
                for( int j=0 ; j < new_w_size ; j++){
                    aux_image[i][j] = aux_data[i][j];
                }
            }
            // send subimage
            ccsds_wavelet_transform_2D(aux_image, new_w_size, new_h_size);
            // copy back image
            for( int i=0 ; i < new_h_size ; i++){
                for( int j=0 ; j < new_w_size ; j++){
                    aux_data[i][j] = aux_image[i][j];
                }
            }
            for(unsigned i = 0; i < new_h_size; i++){
                free(aux_image[i]);
            }
            free(aux_image);
        }	
        
        ++iteration;
    }
    //transform the output image
    for(unsigned int i = 0; i < h_size_padded; i++)
    {			
        for(unsigned int j = 0; j < w_size_padded; j++)		
        {
            if( aux_data[i][j] >= 0)
                transformed_image[i][j] = (int)(aux_data[i][j] + 0.5);
            else 
                transformed_image[i][j] = (int)(aux_data[i][j] -0.5);

        }
    }
	
    // copy the image
    //writeBMP(compression_data->filename_output, transformed_image,*compression_data->w_size,*compression_data->h_size );
    for(unsigned i = 0; i < h_size_padded; i++){
        free(aux_data[i]);
    }
    free(aux_data);
    
}



/**
 * \brief Entry point of the DWT part of the compression.
 */

void dwt2D_compression_computation(
	compression_image_data_t *compression_data,
    int  **image_data,
    int  **transformed_image,
	unsigned int h_size_padded,
	unsigned int w_size_padded
	)
{
    

    if (compression_data->type_of_compression)
	{
        dwt2D_compression_computation_float(compression_data, image_data, transformed_image);
		// Step 1 transform the image 
		/*
		##########################################################################################################
		# This function take the image that has been processed for each of the levels of the DWT 2D and
		# re-arrange the data so each 8 by 8 block contains a family of de DC component been the DC component
		# in 0 0 of that block.
		##########################################################################################################
		*/

		coeff_regroup(transformed_image, h_size_padded, w_size_padded);
	}
	else{
		// integer encoding
        dwt2D_compression_computation_integer(compression_data, image_data, transformed_image);
		coefficient_scaling(transformed_image, h_size_padded, w_size_padded);
	}

}

/**
 * \brief Entry point of the reorder of the blocks.
 */
void coeff_regroup(
    int  **transformed_image,
    unsigned int h_size_padded,
    unsigned int w_size_padded
    )
{

    int  **temp = NULL;
	temp = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		temp[i] = (int *)calloc(w_size_padded, sizeof(int));
	}
	// HH1 band. Starts with grandchildren of family 2
	for (unsigned int i = (h_size_padded>>1); i < h_size_padded; i+=4)
	{
		for (unsigned int j = (w_size_padded>>1); j < w_size_padded; j+=4)
		{
			unsigned int x = ((i - (h_size_padded>>1)) << 1);
			unsigned int y = ((j - (w_size_padded>>1)) << 1);
			for (unsigned int p = 4; p < 8; ++p)
			{
				for (unsigned int k = 4; k < 8; ++k)
				{
					temp[x + p][y + k] = transformed_image[i + p - 4][j + k -4];
				}
			}
		}
	}
	// HL1 Band grandchildren of family 0
	for (unsigned int i = 0; i < (h_size_padded >> 1) ; i+=4)
	{
		for (unsigned int j = (w_size_padded>>1); j < w_size_padded; j+=4)
		{
			unsigned int x = (i  << 1);
			unsigned int y = ((j - (w_size_padded>>1)) << 1);
			for (unsigned int p = 0; p < 4; ++p)
			{
				for (unsigned int k = 4; k < 8; ++k)
				{
					temp[x + p][y + k] = transformed_image[i + p][j + k - 4];
				}
			}
		}
	}
	// LH1 band. grandchildren of family 1
	for (unsigned int i = (h_size_padded>>1); i < h_size_padded; i+=4)
	{
		for (unsigned int j = 0; j < (w_size_padded>>1); j+=4)
		{
			unsigned int x = ((i - (h_size_padded>>1)) << 1);
			unsigned int y = (j<< 1);
			for (unsigned int p = 4; p < 8; ++p)
			{
				for (unsigned int k = 0; k < 4; ++k)
				{
					temp[x + p][y + k] = transformed_image[i + p - 4][j + k];
				}
			}
		}
	}
	// HH2 band. children of family 2
	for (unsigned int i = (h_size_padded>>2); i < (h_size_padded>>1); i+=2)
	{
		for (unsigned int j = (w_size_padded>>2); j < (w_size_padded>>1); j+=2)
		{
			unsigned int x = ((i - (h_size_padded>>2)) <<2);
			unsigned int y = ((j - (w_size_padded>>2)) <<2);
			temp[x + 2][y + 2] =  transformed_image[i][j];			
			temp[x + 2][y + 3] =  transformed_image[i][j + 1];			
			temp[x + 3][y + 2] =  transformed_image[i + 1][j];			
			temp[x + 3][y + 3] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// HL2 band. children of family 0
	for (unsigned int i = 0; i < (h_size_padded>>2); i+=2)
	{
		for (unsigned int j = (w_size_padded>>2); j < (w_size_padded>>1); j+=2)
		{
			unsigned int x = (i <<2);
			unsigned int y = ((j - (w_size_padded>>2)) <<2);
			temp[x][y + 2] =  transformed_image[i][j];			
			temp[x][y + 3] =  transformed_image[i][j + 1];			
			temp[x + 1][y + 2] =  transformed_image[i + 1][j];			
			temp[x + 1][y + 3] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// LH2 band. children of family 1
	for (unsigned int i = (h_size_padded>>2); i < (h_size_padded>>1); i+=2)
	{
		for (unsigned int j = 0; j < (w_size_padded>>2); j+=2)
		{
			unsigned int x = ((i - (h_size_padded>>2)) <<2);
			unsigned int y = (j<<2);
			temp[x + 2][y] =  transformed_image[i][j];			
			temp[x + 2][y + 1] =  transformed_image[i][j + 1];			
			temp[x + 3][y] =  transformed_image[i + 1][j];			
			temp[x + 3][y + 1] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// HH3 band parent family 2
	unsigned int x = (h_size_padded>>3);
	unsigned int y = (w_size_padded>>2);
	for (unsigned int i = (h_size_padded>>3); i < (h_size_padded>>2); ++i)
	{
		for (unsigned int j = (w_size_padded>>3); j < (w_size_padded>>2); ++j)
		{
			temp[((i - x) <<3) + 1][((j - (w_size_padded>>3)) <<3) + 1] = transformed_image[i][j];
		}
	}
	// HL3 band parent family 0
	for (unsigned int i = 0; i < (h_size_padded>>3); ++i)
	{
		for (unsigned int j = (w_size_padded>>3); j < (w_size_padded>>2); ++j)
		{
			temp[i << 3][((j - (w_size_padded>>3)) <<3) + 1] = transformed_image[i][j];
		}
	}
	// LH3 band parent family 1
	for (unsigned int i = (h_size_padded>>3); i < (h_size_padded>>2); ++i)
	{
		for (unsigned int j = 0; j < (w_size_padded>>3); ++j)
		{
			temp[((i - x) <<3) + 1][j<<3] = transformed_image[i][j];
		}
	}
	// LL3 band , DC components
	for (unsigned int i = 0; i < (h_size_padded>>3); ++i)
	{
		for (unsigned int j = 0; j < (w_size_padded>>3); ++j)
		{
			temp[i<<3][j<<3] =  transformed_image[i][j];
		}
	}
	// copy the values to transformed_image
	for(unsigned int i = 0; i < h_size_padded; i++)
	{		
		for(unsigned int j = 0; j < w_size_padded; j++)		
		{
				transformed_image[i][j] = temp[i][j];
		}
	}
	// finish conversion
}


void coefficient_scaling (
	int **transformed_image,
	unsigned int h_size_padded,
    unsigned int w_size_padded
)
{
	// scaling the coefficients
	unsigned short hh_1 = 1;
	unsigned short hl_lh = 2;
	unsigned short hh_2 = 2;
	unsigned short hl_lh_2 = 4;
	unsigned short hh_3 = 4;
	unsigned short hl_lh_3 = 8;
	unsigned short ll_4 = 8;

	// HH1 band. 
	for (unsigned int i = (h_size_padded>>1); i < h_size_padded; i++)
	{
		for(unsigned int j = (w_size_padded>>1); j < w_size_padded; j++)
		{
			transformed_image[i][j] = transformed_image[i][j] * hh_1;
		}
	}
		
	for (unsigned int i = 0; i < (h_size_padded>>1); i++)
	{
		for(unsigned int j = (w_size_padded>>1); j < w_size_padded; j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh;
		}
	}
		
	// LH1
	for (unsigned int i = (h_size_padded>>1); i < h_size_padded; i++)
	{
		for(unsigned int j = 0; j < (w_size_padded>>1); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh;
		}
	}
		

	// HH2 band. 
	for (unsigned int i = (h_size_padded>>2); i < (h_size_padded>>1); i++)
	{
		for(unsigned int j = (w_size_padded>>2); j < (w_size_padded>>1); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hh_2;
		}
	}
		

	// HL2
	for (unsigned int i = 0; i < (h_size_padded>>2); i++)
	{
		for(unsigned int j = (w_size_padded>>2); j < (w_size_padded>>1); j++)
		{	
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh_2;
		}
	}
		
	// LH2

	for (unsigned int i = (h_size_padded>>2); i < (h_size_padded>>1); i++)
	{
		for(unsigned int j = 0; j < (w_size_padded>>2); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh_2;
		}
	}
		
	// HH3 band. 
	for (unsigned int i = (h_size_padded>>3); i < (h_size_padded>>2); i++)
	{
		for(unsigned int j = (w_size_padded>>3); j < (w_size_padded>>2); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hh_3;
		}
	}
		

	// HL3
	for (unsigned int i = 0; i < (h_size_padded>>3); i++)
	{
		for(unsigned int j = (w_size_padded>>3); j < (w_size_padded>>2); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh_3;
		}
	}
		
	// LH3
	for (unsigned int i = (h_size_padded>>3); i < (h_size_padded>>2); i++)
	{
		for(unsigned int j = 0; j < (w_size_padded>>3); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  hl_lh_3;
		}
	}
		

	// LL3
	for (unsigned int i = 0; i < (h_size_padded>>3); i++)
	{
		for(unsigned int j = 0; j < (w_size_padded>>3); j++)
		{
			transformed_image[i][j] = transformed_image[i][j] *  ll_4;
		}	
	}
		
}


void build_block_string(int **transformed_image, unsigned int h_size, unsigned int w_size, int **block_string)
{
	unsigned int block_h = h_size / BLOCKSIZEIMAGE;
	unsigned int block_w = w_size / BLOCKSIZEIMAGE;

	unsigned int total_blocks = block_h * block_w;
	unsigned int counter = 0;
	for (unsigned int i = 0; i < block_h; ++i)
	{
		for (unsigned int j = 0; j < block_w; ++j)
		{
			// outer loop to loop over the blocks
			for (unsigned int x = 0; x < BLOCKSIZEIMAGE; ++x)
			{
				for (unsigned int y =0; y < BLOCKSIZEIMAGE; ++y)
				{
					// this loops is for acces each of the blocks
					block_string[counter][x * BLOCKSIZEIMAGE + y] = transformed_image[i*BLOCKSIZEIMAGE +x][j*BLOCKSIZEIMAGE+y];
				}
				
			}
			++counter;
		}
	}
}

void dc_encoding(
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	int **block_string,
	unsigned int segment_number
	)

{
	int max_ac_value = 0;
	short int quantization_factor = 0;
	unsigned int k = 0; // k is the quantization factor 
	for (unsigned int i = 0; i < compression_data->segment_size; ++i)
	{
		// loop over the blocks to get the dc value
		unsigned int final_pos = i + (segment_number * compression_data->segment_size);
		int dc_value = block_string[final_pos][0];
		// calculate the dc bit size
		unsigned int dc_bit_size = 0;
		if (dc_value < 0)
		{
			// calculate the dc bit size counting the number of bits needed to represent the dc value
			dc_bit_size = ceil(log2(abs(dc_value) + 1));
		}
		else
		{
			// calculate the dc bit size counting the number of bits needed to represent the dc value
			
			dc_bit_size = ceil(log2(dc_value + 1));
			//dc_bit_size = ceil(log2(1 + dc_value)) + 1;
		}
		// if the dc bit size is greater than the max bit size, then set it to the max bit size
		header_data->bit_depth_dc = (unsigned char) std::max(dc_bit_size, (unsigned int)(header_data->bit_depth_dc));
		// check all of the ac values to see if they are greater than the max ac value
		for (unsigned int j = 1; j < BLOCKSIZEIMAGE * BLOCKSIZEIMAGE; ++j)
		{
			if (std::abs(block_string[final_pos][j]) > std::abs(max_ac_value))
			{
				max_ac_value = block_string[final_pos][j];
			}
		}
	}
	// calculate the ac bit size
	header_data->bit_depth_ac = (unsigned char)(std::ceil(std::log2(std::abs(max_ac_value))) + 1) ;
	// finish getting the AC and DC bit sizes
	// get the quantization value
	if (header_data->bit_depth_dc <= 3)
	{
		quantization_factor = 0;
	}
	else if (((header_data->bit_depth_dc - (1 + header_data->bit_depth_ac >> 1)) <= 1) && (header_data->bit_depth_dc > 3) )
	{
		quantization_factor = header_data->bit_depth_dc - 3;
	}
	else if (((header_data->bit_depth_ac - (1 + header_data->bit_depth_ac >> 1)) > 10) && (header_data->bit_depth_dc > 3) )
	{
		quantization_factor = header_data->bit_depth_dc - 10;
	}
	else
	{
		quantization_factor = 1 + (1 + header_data->bit_depth_ac >> 1);
	}

	//shift of the DC component
	k = (1 << quantization_factor) - 1;
	for (unsigned int i = 0; i < compression_data->segment_size; ++i)
	{
		
	}

}


void ac_encoding(
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	int **block_string,
	unsigned int segment_number
	)

{
	
}


void compute_bpe(
    compression_image_data_t *compression_data,
    int **block_string,
    unsigned int total_blocks
    )
{
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
	// create and allocate memory for the header
	header_data_t *header_data = (header_data_t *)malloc(sizeof(header_data_t));
	// for the first header add that is the first header and init the values to 0
	header_data->start_img_flag = true;
	header_data->end_img_flag = false;
	header_data->bit_depth_dc = 1;
	header_data->bit_depth_ac = 1;
	header_data->part_2_flag = false;
	header_data->part_3_flag = false;
	header_data->part_4_flag = false;
	header_data->pad_rows = 0;
	// now loop over the number of segments and calculate the bpe for each segment
	for (unsigned int i = 0; i < num_segments; ++i)
	{
		// update the segment number
		header_data->segment_count = i;
		// now loop over the number of blocks in each segment and calculate the bpe for each block	
		// First calculate DC encoding
		dc_encoding(compression_data,header_data, block_string, i);
		// second calculate AC encoding
		ac_encoding(compression_data,header_data, block_string, i);
		// third update header
			

		
		// check if we are at the last segment and if so add the last header
		if (i == num_segments - 1)
		{
			header_data->end_img_flag = true;
		}
		// write the segment to the binary output with the header
		// write header and clean up the header
		void header_write(header_data_t *header_data, unsigned int segment_number);
	}
	// free the header data
	free(header_data);


}