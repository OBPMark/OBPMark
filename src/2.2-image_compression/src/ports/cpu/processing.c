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

void ac_depth_encoder(block_data_t *block_data,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number);
void ac_gaggle_encode(block_data_t *block_data, compression_image_data_t *compression_data, header_data_t *header_data,unsigned int max_k,unsigned int id_length,unsigned int gaggle_number,unsigned int number_of_gaggles,bool reminder_gaggle,unsigned int N);
void dpcm_ac_mapper(block_data_t *block_data,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int N);

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

unsigned long conv_to_twos_complement(
	unsigned int value,
	unsigned int num_bits
)
{
	unsigned long temp;
	unsigned long complement;
	short i = 0;

	if(num_bits == 1)
	 return 0;

	if((num_bits >= sizeof(unsigned long) * 8) || (num_bits == 0))
	{
		printf("BPE_DATA_ERROR: num_bits is out of range\n");
	}

	if (value >= 0)
		return (unsigned long) value;
	else
	{
		complement = ~(unsigned long) (-value);
		temp = 0;
		for ( i = 0; i < num_bits; i ++)
		{
			temp <<= 1;
			temp ++;
		}
		complement &= temp;
		complement ++;
		return complement;
	}
}


void coding_quantized_coefficients(
	block_data_t *block_data,
	unsigned int num_blocks,
	unsigned int N
)
{
	unsigned long x_min = - (1<<(N-1));
	unsigned long x_max = ((1<<(N-1)) - 1);
	unsigned long theta = 0;
	long gamma = 0;

	block_data[0].mapped_dc = block_data[0].shifted_dc; // reference sample
	for(unsigned int i = 1; i < num_blocks; i ++)
	{
		// calculate theta
		theta = std::min(block_data[i-1].shifted_dc - x_min, x_max + block_data[i-1].shifted_dc);
		// calculate gamma
		gamma = block_data[i].shifted_dc - block_data[i-1].shifted_dc;
		
		if ( 0 <= gamma && gamma <= theta)
		{
			block_data[i].mapped_dc = 2 * gamma;
		}
		else if (-theta <= gamma && gamma < 0)
		{
			block_data[i].mapped_dc = 2 * std::abs(gamma) -1;
		}
		else
		{
			block_data[i].mapped_dc =  theta + std::abs(gamma);
		}
	}

}

void dc_encoder(
	block_data_t *block_data ,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int max_k,
	unsigned int id_length,
	unsigned int gaggle_id,
	unsigned int number_of_gaggles,
	bool reminder_gaggle,
	unsigned int N 
)
{
	const bool brute_force = false;
	unsigned int start_position = 0;
	unsigned int end_position = 0;
	unsigned char encode_selection = 0;

	if (brute_force)
	{
		// brute force coding is not implemented in this version doe to the lack of Header part 3 missin
	}
	else
	{
		//heuristic coding
		int delta = 0;
		unsigned int gaggle_size = GAGGLE_SIZE;
		// encode selection
		encode_selection = 0;
		// calculate the start position of this gaggle
		start_position = gaggle_id * GAGGLE_SIZE;
		end_position = start_position + GAGGLE_SIZE;
		// check if this is the first gaggle, so the start position is 1 instead of 0
		if (gaggle_id == 0)
		{
			start_position = 1;
		}
		// check if this is the last gaggle, so the end position is remaining values instead of GAGGLE_SIZE
		if (reminder_gaggle)
		{
			end_position = compression_data->segment_size;
			gaggle_size = compression_data->segment_size - start_position;
		}
		// sum the delta of all the blocks in this gaggle
		for (unsigned int i = start_position; i < end_position; i ++)
		{
			delta += block_data[i].mapped_dc;
		}

		// table of code options Table 4-10 in the document
		if(64 * delta >= 23 * gaggle_size * (1 <<(N)))
		{
			encode_selection = UNCODED_VALUE;
		}
		else if (207 * gaggle_size > 128 * delta)
		{
			encode_selection = 0;
		}
		else if ((long)(gaggle_size * (1 << (N+5))) <= (long)(128 * delta + 49 * gaggle_size))
		{
			encode_selection = N - 2;
		}
		else
		{
			encode_selection = 0;
			while ((long)(gaggle_size * (1 << (encode_selection + 7 ))) <= (long)(128 * delta + 49 * gaggle_size))
			{
				encode_selection ++;
			}
			encode_selection --; // to adust the value of encode_selection to the correct value
		}

	}


	// print the header 
	// TODO
	// encode_selection with size id_length

	// now print the output values
	for (unsigned int i = start_position; i < start_position + GAGGLE_SIZE; i ++)
	{
		if ((encode_selection == UNCODED_VALUE) || (i == 0) )
		{
			// print the uncoded value. block_data[i].mapped_dc with size N
			//TODO
		}
		else
		{
			// print the coded value.  1 with size (((block_data[i].mapped_dc + 1) >> encode_selection) + 1) // coding first part
			// TODO
		}
	}
	// now generate the second part if the encode_selection is not UNCODED_VALUE
	if (encode_selection != UNCODED_VALUE)
	{
		for (unsigned int i = std::max(start_position, (unsigned int)(1)); i < (start_position + GAGGLE_SIZE); ++i )
		{
			// print the coded value.  block_data[i].mapped_dc with size encode_selection 
			// TODO
		}
	}

}


void dc_entropy_coding(
	block_data_t *block_data ,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int N,
	unsigned int quantization_factor
)
{
	unsigned int id_length = 0;
	unsigned int max_k = 0;
	unsigned int number_of_gaggles = 0;
	bool reminder_gaggle = false;
	
	// get the id_length and max_k base of N
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
	else if (N <= 8)
	{
		max_k = 6;
		id_length = 3;
	}
	else if (N <= 10)
	{
		max_k = 8;
		id_length = 4;
	}
	else {
		printf("BPE_DATA_ERROR: N is out of range\n");
	}
	// calculate the number of gaggles in this segment, each gaggle has a size of GAGGLE_SIZE, if the number of blocks is not divisible by GAGGLE_SIZE, then there will be one more gaggle
	number_of_gaggles = compression_data->segment_size / GAGGLE_SIZE;
	if (compression_data->segment_size % GAGGLE_SIZE != 0)
	{
		number_of_gaggles ++;
		reminder_gaggle = true;
	}
	// loop over the gaggles
	for (unsigned int i = 0; i < number_of_gaggles; i ++)
	{
		dc_encoder(block_data, compression_data, header_data, max_k, id_length, i, number_of_gaggles,reminder_gaggle, N);
	}

	// additional bit planes of DC coefficients
	if (header_data-> bit_depth_ac < quantization_factor)
	{
		unsigned int num_additional_bit_planes = 0;
		if (!compression_data->type_of_compression)
		{
			// integer encoding
			num_additional_bit_planes = quantization_factor - header_data->bit_depth_ac; // missing part 4 of the header data
		}
		else
		{
			// floating point encoding
			num_additional_bit_planes = quantization_factor - header_data->bit_depth_ac;
		}
		// loop over each bit plane in order to decrease significance
		for (unsigned int i = 0; i < num_additional_bit_planes; i++)
		{
			// loop over each block in the segment
			for (unsigned int k = 0; k < compression_data->segment_size; k++)
			{
				// print (block_data[k].dc_reminder >> (quantization_factor - i -1)) with size 1;
				// TODO
			}
		}
	}


}




void dc_encoding(
	compression_image_data_t *compression_data,
	block_data_t *block_data,
	header_data_t *header_data,
	int **block_string,
	unsigned int segment_number
	)

{
	int max_ac_value = 0;
	short int quantization_factor = 0;
	unsigned int k = 0; // k is the quantization factor 
	// create a new array of block_data_t to store the DC values
	

	for (unsigned int i = 0; i < compression_data->segment_size; ++i)
	{
		int max_ac_value_block = 0;
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
			if (std::abs(block_string[final_pos][j]) > std::abs(max_ac_value_block))
			{
				max_ac_value_block = block_string[final_pos][j];
			}
		}
	block_data[i].max_ac_bit_size = max_ac_value_block;
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
		unsigned int final_pos = i + (segment_number * compression_data->segment_size);
		unsigned long new_value_twos = conv_to_twos_complement(block_string[final_pos][0], header_data->bit_depth_dc);
		block_data[i].shifted_dc = new_value_twos >> quantization_factor;
		block_data[i].dc_reminder = (unsigned short)(new_value_twos & k);
	}

	// get the number of bits needed to represent the quantized dc values
	unsigned int N = std::max(header_data->bit_depth_dc - quantization_factor, 1);

	// maximum possible value of N is 10

	// case for N = 1
	if (N == 1)
	{
		for (unsigned int i = 0; i < compression_data->segment_size; ++i)
		{
			// write  block_data[i].shifted_dc with only one bit
			// TODO
		}
	}
	else
	{
		coding_quantized_coefficients(block_data, compression_data->segment_size, N);
		dc_entropy_coding(block_data, compression_data, header_data, N, quantization_factor);
	}

	

}


void ac_encoding(
	compression_image_data_t *compression_data,
	block_data_t *block_data,
	header_data_t *header_data,
	int **block_string,
	unsigned int segment_number
	)

{
	unsigned char bit_plane = 0;
	// check if the ac bit depth is bigger that 0
	if (header_data->bit_depth_ac > 0) //if not not need to be coded
	{
		// if the ac bit depth is 1, the codiffication is binary 
		if (header_data -> bit_depth_ac == 1)
		{
			for (unsigned int i = 0; i < compression_data->segment_size; ++i)
			{
				// write block_data[i].max_ac_bit_size[j] with only one bit
				// TODO
				
			}
			
		}
		else
		{
			
			ac_depth_encoder(block_data, compression_data, header_data, segment_number);
		}

		for ( bit_plane = header_data-> bit_depth_ac; bit_plane > 0; bit_plane --)
		{
			// stage 0 to stage 3
			bit_plane_encoding(block_data, block_string, compression_data, header_data, segment_number, bit_plane);
			// stage 4

		}
	} 
	
}

void ac_depth_encoder(
	block_data_t *block_data,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number
	)
{
	unsigned int N = 0;
	unsigned int id_length = 0;
	unsigned int max_k = 0;
	unsigned int number_of_gaggles = 0;
	bool reminder_gaggle = false;

	// calculate N
	while(header_data->bit_depth_ac >> N > 0)
	{
		N++;
	}
	
	dpcm_ac_mapper(block_data, compression_data, header_data, segment_number, N);

	// calculate id_length
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
	else if (N <= 5)
	{
		max_k = 6;
		id_length = 3;
	}
	else
	{
		printf("N is too big in AC encoding\n");
		max_k = 0;
		id_length = 0;
	}

	// gaggle encoding
	// calculate the number of gaggles in this segment, each gaggle has a size of GAGGLE_SIZE, if the number of blocks is not divisible by GAGGLE_SIZE, then there will be one more gaggle
	number_of_gaggles = compression_data->segment_size / GAGGLE_SIZE;
	if (compression_data->segment_size % GAGGLE_SIZE != 0)
	{
		number_of_gaggles ++;
		reminder_gaggle = true;
	}
	// loop over the gaggles
	for (unsigned int i = 0; i < number_of_gaggles; i ++)
	{
		ac_gaggle_encode(block_data, compression_data, header_data, max_k, id_length, i, number_of_gaggles,reminder_gaggle, N);
	}

}

void ac_gaggle_encode(
	block_data_t *block_data,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int max_k,
	unsigned int id_length,
	unsigned int gaggle_number,
	unsigned int number_of_gaggles,
	bool reminder_gaggle,
	unsigned int N
	)
{
	const bool brute_force = false;
	unsigned int start_position = 0;
	unsigned int end_position = 0;
	unsigned char encode_selection = 0;

	if (brute_force)
	{
		// brute force coding is not implemented in this version doe to the lack of Header part 3 missin
	}
	else
	{
		//heuristic coding
		int delta = 0;
		unsigned int gaggle_size = GAGGLE_SIZE;
		// encode selection
		encode_selection = 0;
		// calculate the start position of this gaggle
		start_position = gaggle_number * GAGGLE_SIZE;
		end_position = start_position + GAGGLE_SIZE;
		// check if this is the first gaggle, so the start position is 1 instead of 0
		if (gaggle_number == 0)
		{
			start_position = 1;
		}
		// check if this is the last gaggle, so the end position is remaining values instead of GAGGLE_SIZE
		if (reminder_gaggle)
		{
			end_position = compression_data->segment_size;
			gaggle_size = compression_data->segment_size - start_position;
		}
		// sum the delta of all the blocks in this gaggle
		for (unsigned int i = start_position; i < end_position; i ++)
		{
			delta += block_data[i].mapped_ac;
		}

		// table of code options Table 4-10 in the document
		if(64 * delta >= 23 * gaggle_size * (1 <<(N)))
		{
			encode_selection = UNCODED_VALUE;
		}
		else if (207 * gaggle_size > 128 * delta)
		{
			encode_selection = 0;
		}
		else if ((long)(gaggle_size * (1 << (N+5))) <= (long)(128 * delta + 49 * gaggle_size))
		{
			encode_selection = N - 2;
		}
		else
		{
			encode_selection = 0;
			while ((long)(gaggle_size * (1 << (encode_selection + 7 ))) <= (long)(128 * delta + 49 * gaggle_size))
			{
				encode_selection ++;
			}
			encode_selection --; // to adust the value of encode_selection to the correct value
		}

	}


	// print the header 
	// TODO
	// encode_selection with size id_length

	// now print the output values
	for (unsigned int i = start_position; i < start_position + GAGGLE_SIZE; i ++)
	{
		if ((encode_selection == UNCODED_VALUE) || (i == 0) )
		{
			// print the uncoded value. block_data[i].mapped_ac with size N
			//TODO
		}
		else
		{
			// print the coded value.  1 with size (((block_data[i].mapped_ac + 1) >> encode_selection) + 1) // coding first part
			// TODO
		}
	}
	// now generate the second part if the encode_selection is not UNCODED_VALUE
	if (encode_selection != UNCODED_VALUE)
	{
		for (unsigned int i = std::max(start_position, (unsigned int)(1)); i < (start_position + GAGGLE_SIZE); ++i )
		{
			// print the coded value.  block_data[i].mapped_ac with size encode_selection 
			// TODO
		}
	}
}


void dpcm_ac_mapper(
	block_data_t *block_data,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int N
	)
{
	unsigned long x_min = 0;
	unsigned long x_max = ((1<<(N-1)) - 1);
	unsigned long theta = 0;
	long gamma = 0;

	block_data[0].mapped_ac = block_data[0].max_ac_bit_size; // reference sample
	for(unsigned int i = 1; i < compression_data->segment_size; i ++)
	{
		// calculate theta
		theta = std::min(block_data[i-1].max_ac_bit_size - x_min, x_max + block_data[i-1].max_ac_bit_size);
		// calculate gamma
		gamma = block_data[i].max_ac_bit_size - block_data[i-1].max_ac_bit_size;
		
		if ( 0 <= gamma && gamma <= theta)
		{
			block_data[i].mapped_ac = 2 * gamma;
		}
		else if (-theta <= gamma && gamma < 0)
		{
			block_data[i].mapped_ac = 2 * std::abs(gamma) -1;
		}
		else
		{
			block_data[i].mapped_ac =  theta + std::abs(gamma);
		}
	}

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
		block_data_t *block_data = (block_data_t *)malloc(sizeof(block_data_t) * compression_data->segment_size);
		// First calculate DC encoding
		dc_encoding(compression_data,block_data,header_data, block_string, i);
		// second calculate AC encoding
		ac_encoding(compression_data,block_data,header_data, block_string, i);
		// third update header
			

		
		// check if we are at the last segment and if so add the last header
		if (i == num_segments - 1)
		{
			header_data->end_img_flag = true;
		}
		// write the segment to the binary output with the header
		// write header and clean up the header
		//void header_write(header_data_t *header_data, unsigned int segment_number);
		// free the block_data array
		free(block_data);
	}
	// free the header data
	free(header_data);


}


void bit_plane_encoding(
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number
	)
{
	
	
}
