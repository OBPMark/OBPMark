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
void ac_gaggle_encode(block_data_t *block_data, compression_image_data_t *compression_data, header_data_t *header_data,unsigned int max_k,unsigned int id_length,unsigned int gaggle_number,unsigned int number_of_gaggles,bool reminder_gaggle,unsigned int N, unsigned int segment_number);
void dpcm_ac_mapper(block_data_t *block_data,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int N);
void bit_plane_encoding(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number);
void stages_encoding(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number);
void coding_options(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number,unsigned int block_index,unsigned int blocks_in_gaggle,unsigned char *code_option_gaggle,bool *hit_flag);
void gaggle_encode_1(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number,unsigned int block_index,unsigned int blocks_in_gaggle,unsigned char *code_option_gaggle,bool *hit_flag);
void gaggle_encode_2(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number,unsigned int block_index,unsigned int blocks_in_gaggle,unsigned char *code_option_gaggle,bool *hit_flag);
void gaggle_encode_3(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number,unsigned int block_index,unsigned int blocks_in_gaggle,unsigned char *code_option_gaggle,bool *hit_flag);
void ref_bit_end_encode(block_data_t *block_data,int **block_string,compression_image_data_t *compression_data,header_data_t *header_data,unsigned int segment_number,unsigned int bit_plane_number);


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
	//unsigned int y = (w_size_padded>>2);
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

	//unsigned int total_blocks = block_h * block_w;
	//unsigned int counter = 0;

	#pragma omp parallel for
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
					block_string[i * block_w + j][x * BLOCKSIZEIMAGE + y] = transformed_image[i*BLOCKSIZEIMAGE +x][j*BLOCKSIZEIMAGE+y];
					//block_string[counter][y] = transformed_image[i*BLOCKSIZEIMAGE +x][j*BLOCKSIZEIMAGE+y];
				}
				//++counter;
			}
			//++counter;
		}
	}
}

unsigned long conv_to_twos_complement(
	int value,
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
	long x_min = - (1<<(N-1));
	long x_max = ((1<<(N-1)) - 1);
	unsigned int number_bits = 0;
	long theta = 0;
	long gamma = 0;

	block_data[0].mapped_dc = block_data[0].shifted_dc; // reference sample

	for (unsigned int i = 0; i < (N - 1); i++)
	{
		number_bits = (number_bits<<1) +1;
	}

	if ((block_data[0].shifted_dc & (1 << (N - 1))) > 0)
	{
		block_data[0].shifted_dc = - (short) (((block_data[0].shifted_dc ^ number_bits) & number_bits) + 1);
	}

	for(unsigned int i = 1; i < num_blocks; i ++)
	{	
		if ((block_data[i].shifted_dc & (1 << (N - 1))) > 0)
		{
			block_data[i].shifted_dc = - (short) (((block_data[i].shifted_dc ^ number_bits) & number_bits) + 1);
		}
		// calculate theta
		theta = std::min(block_data[i-1].shifted_dc - x_min, x_max - block_data[i-1].shifted_dc);
		// calculate gamma
		gamma = block_data[i].shifted_dc - block_data[i-1].shifted_dc;
		//printf("theta = %ld: %ld %ld %ld\n", theta, block_data[i-1].shifted_dc, x_min, x_max);
		//printf("gamma = %ld\n", gamma);
		if ( 0 <= gamma && gamma <= theta)
		{
			block_data[i].mapped_dc = 2 * gamma;
			//printf("%ld\n", block_data[i].mapped_dc);
		}
		else if (-theta <= gamma && gamma < 0)
		{
			block_data[i].mapped_dc = 2 * std::abs(gamma) -1;
			//printf("%ld\n", block_data[i].mapped_dc);
		}
		else
		{
			block_data[i].mapped_dc =  theta + std::abs(gamma);
			//printf("%ld\n", block_data[i].mapped_dc);
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
	unsigned int N ,
	unsigned int segment_number
)
{
	const bool brute_force = true;
	unsigned int start_position = 0;
	unsigned int end_position = 0;
	unsigned char encode_selection = 0;
	unsigned int total_bits = 0;
	unsigned int min_bits = 0XFFFF;

	if (brute_force)
	{
		// brute force coding is not implemented in this version doe to the lack of Header part 3 missing
		unsigned int gaggle_size = GAGGLE_SIZE;
		start_position = gaggle_id * GAGGLE_SIZE;
		end_position = start_position + GAGGLE_SIZE;
		// check if this is the first gaggle, so the start position is 1 instead of 0
		// check if this is the last gaggle, so the end position is remaining values instead of GAGGLE_SIZE
		if (reminder_gaggle)
		{
			end_position = compression_data->segment_size;
			gaggle_size = compression_data->segment_size - start_position;
		}
		encode_selection = UNCODED_VALUE;
		for (unsigned int k = 0; k <= max_k; ++k)
		{
			//printf("StartIndex %d\n", start_position);
			if (start_position == 0)
			{
				total_bits = N;
			}
			else
			{
				total_bits = 0;
			}
			
			for (unsigned int i = std::max(start_position,(unsigned int)1); i < end_position; i ++)
			{
				//printf("Mapped DC %ld %d\n", block_data[i].mapped_dc, total_bits);
				total_bits += ((block_data[i].mapped_dc >> k) + 1) + k;
				
			}
			//printf("%d\n", total_bits);
			if ((total_bits < min_bits) && (total_bits < N * gaggle_size))
			{
				min_bits = total_bits;
				encode_selection = k;
			}
		}
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
		// check if this is the last gaggle, so the end position is remaining values instead of GAGGLE_SIZE
		if (reminder_gaggle)
		{
			end_position = compression_data->segment_size;
			gaggle_size = compression_data->segment_size - start_position;
		}
		// sum the delta of all the blocks in this gaggle
		for (unsigned int i = std::max(start_position,(unsigned int)1); i < end_position; i ++)
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


	//printf("DC: min_k = %d\n", encode_selection);
	// encode_selection with size id_length
	write_to_the_output_segment(compression_data->segment_list,encode_selection,id_length,segment_number);

	// now print the output values
	for (unsigned int i = start_position; i < start_position + GAGGLE_SIZE; i ++)
	{
		if ((encode_selection == UNCODED_VALUE) || (i == 0) )
		{
			write_to_the_output_segment(compression_data->segment_list,block_data[i].mapped_dc,N,segment_number);
		}
		else
		{
			
			write_to_the_output_segment(compression_data->segment_list,1, ((block_data[i].mapped_dc) >> encode_selection)+1 ,segment_number);
		}
	}
	// now generate the second part if the encode_selection is not UNCODED_VALUE
	if (encode_selection != UNCODED_VALUE)
	{
		for (unsigned int i = std::max(start_position, (unsigned int)(1)); i < (start_position + GAGGLE_SIZE); ++i )
		{
			write_to_the_output_segment(compression_data->segment_list,block_data[i].mapped_dc,encode_selection,segment_number);
		}
	}

}


void dc_entropy_coding(
	block_data_t *block_data ,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int N,
	unsigned int quantization_factor,
	unsigned int segment_number
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
		dc_encoder(block_data, compression_data, header_data, max_k, id_length, i, number_of_gaggles,reminder_gaggle, N, segment_number);
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
				write_to_the_output_segment(compression_data->segment_list,(block_data[k].dc_reminder >> (quantization_factor - i - 1)),1,segment_number);
			}
		}
	}


}


void header_output(
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_id)
{
	#ifdef INCLUDE_HEADER
	write_to_the_output_segment(compression_data->segment_list,(int)header_data->start_img_flag,1, segment_id);
	write_to_the_output_segment(compression_data->segment_list,(int)header_data->end_img_flag,1, segment_id);
	write_to_the_output_segment(compression_data->segment_list,header_data->segment_count,8, segment_id);
	write_to_the_output_segment(compression_data->segment_list,header_data->bit_depth_dc,5, segment_id);
	write_to_the_output_segment(compression_data->segment_list,header_data->bit_depth_ac,5, segment_id);
	write_to_the_output_segment(compression_data->segment_list,0,1, segment_id); //RESERVED 
	write_to_the_output_segment(compression_data->segment_list,(int)header_data->part_2_flag,1, segment_id);
	write_to_the_output_segment(compression_data->segment_list,(int)header_data->part_3_flag,1, segment_id);
	write_to_the_output_segment(compression_data->segment_list,(int)header_data->part_4_flag,1, segment_id);

	if (header_data->end_img_flag == true)
	{
		write_to_the_output_segment(compression_data->segment_list,header_data->pad_rows,3, segment_id);
		write_to_the_output_segment(compression_data->segment_list,0,5, segment_id); // RESERVED
	}
	// MISSING PART 2 3 and 4
	#endif
}

short int dc_encoding(
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
		long max_ac_value_block = 0;
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
			
			//dc_bit_size = ceil(log2(dc_value + 1));
			dc_bit_size = ceil(log2(1 + dc_value)) + 1;
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
		//printf("Max AC Segment = %ld\n", std::abs(max_ac_value_block));
		block_data[i].max_ac_bit_size = (unsigned char)(std::ceil(std::log2(std::abs(max_ac_value_block)))) ;
		// if max_ac_value_block is a power of 2, then max_ac_bit_size + 1
		if (ceil(log2(std::abs(max_ac_value_block))) == floor(log2(std::abs(max_ac_value_block))))
		{
			block_data[i].max_ac_bit_size = block_data[i].max_ac_bit_size + 1;
		}
		//printf("max ac bit size %ld\n", block_data[i].max_ac_bit_size);
	}
	// calculate the ac bit size
	header_data->bit_depth_ac = (unsigned char)(std::ceil(std::log2(std::abs(max_ac_value)))) ;
	// if max_ac_value is a power of 2, then max_ac_bit_size + 1
	if (ceil(log2(std::abs(max_ac_value))) == floor(log2(std::abs(max_ac_value))))
	{
		header_data->bit_depth_ac = header_data->bit_depth_ac + 1;
	}
	// finish getting the AC and DC bit sizes
	header_output(compression_data, header_data,segment_number);

	// get the quantization value
	if (header_data->bit_depth_dc <= 3)
	{
		quantization_factor = 0;
	}
	else if (((header_data->bit_depth_dc - (1 + header_data->bit_depth_ac >> 1)) <= 1) && (header_data->bit_depth_dc > 3) )
	{
		quantization_factor = header_data->bit_depth_dc - 3;
	}
	else if (((header_data->bit_depth_dc - (1 + header_data->bit_depth_ac >> 1)) > 10) && (header_data->bit_depth_dc > 3) )
	{
		quantization_factor = header_data->bit_depth_dc - 10;
	}
	else
	{
		quantization_factor = 1 + (header_data->bit_depth_ac >> 1);
	}
	//printf("Quantization factor: %d %d %d\n",quantization_factor, header_data->bit_depth_dc, header_data->bit_depth_ac);
	//shift of the DC component
	k = (1 << quantization_factor) - 1;
	for (unsigned int i = 0; i < compression_data->segment_size; ++i)
	{
		unsigned int final_pos = i + (segment_number * compression_data->segment_size);
		unsigned long new_value_twos = conv_to_twos_complement(block_string[final_pos][0], header_data->bit_depth_dc);
		//printf("ConvTwosComp: %lu %d\n", new_value_twos, quantization_factor);
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
			write_to_the_output_segment(compression_data->segment_list,block_data[i].shifted_dc,1, segment_number);
		}
	}
	else
	{
		
		coding_quantized_coefficients(block_data, compression_data->segment_size, N);
		dc_entropy_coding(block_data, compression_data, header_data, N, quantization_factor,segment_number);
	}
	return quantization_factor;
	

}


void ac_encoding(
	compression_image_data_t *compression_data,
	block_data_t *block_data,
	header_data_t *header_data,
	int **block_string,
	unsigned int segment_number,
	short int quantization_factor
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
				write_to_the_output_segment(compression_data->segment_list,block_data[i].max_ac_bit_size,1, segment_number);
				
			}
			
		}
		else
		{
			
			ac_depth_encoder(block_data, compression_data, header_data, segment_number);
		}

		for ( bit_plane = header_data-> bit_depth_ac; bit_plane > 0; bit_plane --)
		{
			
			if( (bit_plane <= quantization_factor) && ((compression_data->type_of_compression) || ((quantization_factor > BITLL3 ) && (BITLL3 < bit_plane)) ))
			{
				//printf("printing remaining bits\n");
				for (unsigned int i = 0; i < compression_data->segment_size; ++i)
				{
					write_to_the_output_segment(compression_data->segment_list,( (block_data[i].dc_reminder >> (bit_plane - 1)) & 0x01),1, segment_number);
				}
				//printf("end printing remaining bits\n");
			}
			// stage 0 
			// MISSING heather part 4
			//1,2, 3
			bit_plane_encoding(block_data, block_string, compression_data, header_data, segment_number, bit_plane);
			// stage 4
			stages_encoding(block_data, block_string, compression_data, header_data, segment_number, bit_plane);
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
		ac_gaggle_encode(block_data, compression_data, header_data, max_k, id_length, i, number_of_gaggles,reminder_gaggle, N, segment_number);
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
	unsigned int N,
	unsigned int segment_number
	)
{
	const bool brute_force = true;
	unsigned int start_position = 0;
	unsigned int end_position = 0;
	unsigned char encode_selection = 0;

	if (brute_force)
	{
		// brute force coding is not implemented in this version doe to the lack of Header part 3 missing
		// brute force coding is not implemented in this version doe to the lack of Header part 3 missing
		unsigned int gaggle_size = GAGGLE_SIZE;
		start_position = gaggle_number * GAGGLE_SIZE;
		end_position = start_position + GAGGLE_SIZE;
		// check if this is the first gaggle, so the start position is 1 instead of 0
		// check if this is the last gaggle, so the end position is remaining values instead of GAGGLE_SIZE
		if (reminder_gaggle)
		{
			end_position = compression_data->segment_size;
			gaggle_size = compression_data->segment_size - start_position;
		}
		encode_selection = UNCODED_VALUE;
		int total_bits = 0;
		int min_bits = 0XFFFF;
		for (unsigned int k = 0; k <= max_k; ++k)
		{
			if (start_position == 0)
			{
				total_bits = N;
			}
			else
			{
				total_bits = 0;
			}
			
			for (unsigned int i = std::max(start_position,(unsigned int)1); i < end_position; i ++)
			{
				total_bits += ((block_data[i].mapped_ac >> k) + 1) + k;
				
			}
			if ((total_bits < min_bits) && (total_bits < N * gaggle_size))
			{
				min_bits = total_bits;
				encode_selection = k;
			}
		}
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
	// encode_selection with size id_length
	write_to_the_output_segment(compression_data->segment_list,encode_selection,id_length,segment_number);

	// now print the output values
	for (unsigned int i = start_position; i < start_position + GAGGLE_SIZE; i ++)
	{
		if ((encode_selection == UNCODED_VALUE) || (i == 0) )
		{
			// print the uncoded value. block_data[i].mapped_ac with size N
			write_to_the_output_segment(compression_data->segment_list,block_data[i].mapped_ac,N,segment_number);
		}
		else
		{
			// print the coded value.  1 with size (((block_data[i].mapped_ac + 1) >> encode_selection) + 1) // coding first part
			write_to_the_output_segment(compression_data->segment_list,1,(((block_data[i].mapped_ac) >> encode_selection) + 1),segment_number);
		}
	}
	// now generate the second part if the encode_selection is not UNCODED_VALUE
	if (encode_selection != UNCODED_VALUE)
	{
		for (unsigned int i = std::max(start_position, (unsigned int)(1)); i < (start_position + GAGGLE_SIZE); ++i )
		{
			// print the coded value.  block_data[i].mapped_ac with size encode_selection 
			write_to_the_output_segment(compression_data->segment_list,block_data[i].mapped_ac,encode_selection,segment_number);
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
	unsigned long x_max = ((1 << N) - 1);
	long theta = 0;
	short gamma = 0;

	block_data[0].mapped_ac = block_data[0].max_ac_bit_size; // reference sample
	for(unsigned int i = 1; i < compression_data->segment_size; i ++)
	{
		// calculate theta
		theta = std::min(block_data[i-1].max_ac_bit_size - x_min, x_max - block_data[i-1].max_ac_bit_size);
		// calculate gamma
		gamma = block_data[i].max_ac_bit_size - block_data[i-1].max_ac_bit_size;
		//printf("theta AC= %ld: %ld %ld %ld\n", theta, block_data[i-1].max_ac_bit_size, x_min, x_max);
		//printf("gamma AC= %d\n", gamma);
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


void init_block_data(block_data_t *block_data)
{
	// clean the block_data structure
	block_data->shifted_dc =  0;
	block_data->dc_reminder = 0;
	block_data->mapped_dc = 0;
	block_data->max_ac_bit_size = 0;
	block_data->mapped_ac = 0;
	block_data->type_p = 0;
	block_data->tran_b = 0;
	block_data->tran_d = 0;
	block_data->tran_gi = 0;
	block_data->parent_ref_symbol = 0;
	block_data->parent_sym_len = 0;
	block_data->children_ref_symbol = 0;
	block_data->children_sym_len = 0;

	for (unsigned int i = 0; i < 3; ++i)
	{
		block_data->type_ci[i] = 0;
		block_data->tran_hi[i] = 0;
		block_data->grand_children_ref_symbol[i] = 0;
		block_data->grand_children_sym_len[i] = 0;

	}

	for (unsigned int i = 0; i < 12; ++i)
	{
		block_data->type_hi[i] = 0;
	}

	for(unsigned int i = 0; i < MAX_SYMBOLS_IN_BLOCK; ++ i)
	{
		block_data->symbol_block[i].symbol_val = 0;
		block_data->symbol_block[i].symbol_len = 0;
		block_data->symbol_block[i].symbol_mapped_pattern = 0;
		block_data->symbol_block[i].sign = 0;
		block_data->symbol_block[i].type = 0;
	}

}


void compute_bpe(
    compression_image_data_t *compression_data,
    int **block_string,
    unsigned int num_segments
    )
{
 	
	
	
	// now loop over the number of segments and calculate the bpe for each segment
	#pragma omp parallel for
	for (unsigned int i = 0; i < num_segments; ++i)
	{

		// create and allocate memory for the header
		header_data_t *header_data = (header_data_t *)malloc(sizeof(header_data_t));
		// for the first header add that is the first header and init the values to 0
		header_data->start_img_flag = false;
		header_data->end_img_flag = false;
		header_data->bit_depth_dc = 0;
		header_data->bit_depth_ac = 0;
		header_data->part_2_flag = false;
		header_data->part_3_flag = false;
		header_data->part_4_flag = false;
		header_data->pad_rows = 0;
		short int quantization_factor = 0;
		// update the segment number
		header_data->segment_count = i;
		//printf("SEGMENT NUMBER: %d\n", i);
		// check if we are at the last segment and if so add the last header
		if (i == 0)
		{
			header_data->start_img_flag = true;
		}
		if (i == num_segments - 1)
		{
			header_data->end_img_flag = true;
		}
		// now loop over the number of blocks in each segment and calculate the bpe for each block	
		//block_data_t *block_data = (block_data_t *)malloc(sizeof(block_data_t) * compression_data->segment_size);
		block_data_t *block_data = (block_data_t *)calloc(compression_data->segment_size, sizeof(block_data_t));
		//init_block_data(block_data);
		// First calculate DC encoding
		quantization_factor = dc_encoding(compression_data,block_data,header_data, block_string, i);
		// second calculate AC encoding
		ac_encoding(compression_data,block_data,header_data, block_string, i, quantization_factor);
		// third update header
		header_data->start_img_flag = false;
		header_data->bit_depth_dc = 0;
		header_data->bit_depth_ac = 0;
		header_data->part_2_flag = false;
		header_data->part_3_flag = false;
		header_data->part_4_flag = false;
		
		// write the segment to the binary output with the header
		// write header and clean up the header
		//void header_write(header_data_t *header_data, unsigned int segment_number);
		// flush the segment to have byte size
		round_up_last_byte(compression_data->segment_list,i);
		// free the block_data array
		free(block_data);
		
		// free the header data
		free(header_data);
	}



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
	//printf("BitPlane: %d\n", bit_plane_number);
	unsigned int temp_x = 0;
	unsigned int  temp_y = 0;
	unsigned char symbol = 0;
	int temp_value = 0;
	unsigned int start_position = segment_number * compression_data->segment_size;
	unsigned int final_position = 0;
	int bit_set_plane = (1 << (bit_plane_number - 1));
	//printf("Bit_Set_Plane: %d\n", bit_set_plane);

	// loop over the blocks in the segment
	for (int block_num = 0; block_num < compression_data->segment_size; ++block_num)
	{	
		symbol = 0;
		final_position = start_position + block_num;
		// check if something to code
		if (block_data[block_num].max_ac_bit_size < bit_plane_number)
		{
			continue;
		}

		// init block data variables
		//block_data[i].type_p = 0;


		// check the parents of the block_string
		for (int j = 0; j < 3; ++j)
		{
			// missing  Header part 4

			temp_x = final_position;
			// get temp_y for each parent of the block_string, the first parent is in position 1, 
			// the second in position BLOCKSIZEIMAGE and the third in position BLOCKSIZEIMAGE + 1

			switch (j)
			{
			case 0: temp_y = 1; break;
			case 1: temp_y = BLOCKSIZEIMAGE; break;
			case 2: temp_y = BLOCKSIZEIMAGE + 1; break;
			default: temp_y = 0; break;
			}
			if (!compression_data->type_of_compression) // integer encoding
			{
				if( ((j == 0) && (BITHL3 >= bit_plane_number )) || ((j == 1) && (BITLH3 >= bit_plane_number)) || ((j == 2) && (BITHH3>= bit_plane_number)) )
				{
					continue;
				}
			}

			//printf("TYPE P %d %d\n", block_data[block_num].type_p, block_num);
			if ((block_data[block_num].type_p & (1 << (2-j))) == 0 )
			{
				block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_P;
				block_data[block_num].symbol_block[symbol].symbol_len ++;
				block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
				//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
				//printf("SYMBOL 1-1\n");

				//printf("BLOCK STRING VALUE %d bit_set_plane %d\n", block_string[temp_x][temp_y], bit_plane_number);
				if (ABSOLUTE(block_string[temp_x][temp_y]) >= (1 << (bit_plane_number - 1)) &&
					ABSOLUTE(block_string[temp_x][temp_y]) < (1 << bit_plane_number))
				{
					block_data[block_num].type_p += (1 << (2 - j));
					block_data[block_num].symbol_block[symbol].symbol_val += 1;
					block_data[block_num].symbol_block[symbol].sign <<= 1;
					block_data[block_num].symbol_block[symbol].sign += SIGN(block_string[temp_x][temp_y]);
					//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
					//printf("TYPE P POST %d: %d\n", block_data[block_num].type_p, (1 << (2 - j)));
					//printf("SYMBOL 1-2\n");
				}

			}
			else
			{
				temp_value = ((ABSOLUTE(block_string[temp_x][temp_y]))  & (bit_set_plane) ) > 0 ? 1 : 0;
				if (!compression_data->type_of_compression) // integer compression
				{
					// header part 4 custom 2 bits TODO CHECK

					if( ((j == 0) && (BITHL3 < bit_plane_number )) || ((j == 1) && (BITLH3 < bit_plane_number)) || ((j == 2) && (BITHH3 < bit_plane_number)) )
					{
						block_data[block_num].parent_ref_symbol <<= 1;
						block_data[block_num].parent_ref_symbol += temp_value;
						block_data[block_num].parent_sym_len ++;
					}
					
				}
				else
				{
					// float compression
					block_data[block_num].parent_ref_symbol <<= 1;
					block_data[block_num].parent_ref_symbol += temp_value;
					block_data[block_num].parent_sym_len ++;
				}
			}
		}
		// check the children of the block_string
		if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
		{
			++symbol;
		}
		// now determine first part TranB
		//printf("TRAN B1 %d\n", block_data[block_num].tran_b);
		if (block_data[block_num].tran_b == 0)
		{
			bool break_flag = false;
			for ( unsigned int k = 0; k < 3; k++)
			{
				// Header part 4 custom 2 bits
				//
				temp_x = (k >= 1 ? 1 : 0);
				temp_x *= 2;
				temp_y = (k != 1 ? 1 : 0);
				temp_y *= 2;
				if (!compression_data->type_of_compression) // integer compression
				{
					if( ((k == 0) && (BITHL1 >= bit_plane_number )) || ((k == 1) && (BITLH1 >= bit_plane_number)) || ((k == 2) && (BITHH1 >= bit_plane_number)) )
					{
						continue;
					}
				}
				for (unsigned int i = temp_x ; i < temp_x + 2; ++i)
				{
					for (unsigned int j = temp_y; j < temp_y + 2; ++j)
					{
						if ((bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
						{
							block_data[block_num].tran_b = 1;
							block_data[block_num].symbol_block[symbol].symbol_len = 1;
							block_data[block_num].symbol_block[symbol].symbol_val = 1;
							//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
							block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_B;
							//printf("SYMBOL 2-1\n");
							symbol ++;
							// goto DS_UPDATE;
							// exit the loops
							i = temp_x + 2;
							j = temp_y + 2;
							k = 3;
							break_flag = true;
						}
					
					}
				}
				if (!break_flag)
				{
					if (!compression_data->type_of_compression) // integer compression
					{
						if( ((k == 0) && (BITHL1 >= bit_plane_number )) || ((k == 1) && (BITLH1 >= bit_plane_number)) || ((k == 2) && (BITHH1 >= bit_plane_number)) )
						{
							continue;
						}
					}
					temp_x = (k >= 1 ? 1 : 0);
					temp_x *= 4;
					temp_y = (k != 1 ? 1 : 0);
					temp_y *= 4;

					for (unsigned int i = temp_x; i < temp_x + 4; ++i )
					{
						for (unsigned int j = temp_y; j < temp_y + 4; ++j)
						{
							if ((bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
							{
								block_data[block_num].tran_b = 1;
								block_data[block_num].symbol_block[symbol].symbol_len = 1;
								block_data[block_num].symbol_block[symbol].symbol_val = 1;
								//printf("SYMBOL 2-2\n");
								//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
								block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_B;
								
								symbol ++;
								// goto DS_UPDATE;
								// exit the loops
								i = temp_x + 4;
								j = temp_y + 4;
								k = 3;
								break_flag = true;
							}
						}
					}	
				}
			}
		}

		//printf("TRAN B2 %d\n", block_data[block_num].tran_b);
		if (block_data[block_num].tran_b == 0)
		{
			block_data[block_num].symbol_block[symbol].symbol_len = 1;
			block_data[block_num].symbol_block[symbol].symbol_val = 0;
			//printf("SYMBOL 3-1\n");
			//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
			block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_B;
			continue;
		}

		// tran_b  end

		if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
		{
			++symbol;
		}

		if (block_data[block_num].tran_b == 1) // continue scan
		{
			for ( int k = 0; k < 3; k++)
			{
				
				if (!compression_data->type_of_compression) // integer compression
				{
					if( ((k == 0) && (BITHL2 >= bit_plane_number ) && (BITHL1 >= bit_plane_number) ) 
					|| ((k == 1) && (BITLH2 >= bit_plane_number) && (BITLH1 >= bit_plane_number)) 
					|| ((k == 2) && (BITHH2 >= bit_plane_number) && (BITHH1 >= bit_plane_number)) )
					{
						continue;
					}
				}
				
				bool break_flag = false;
				// Header part 4 custom 2 bits
				// 
				if ((block_data[block_num].tran_d & (1 << (2 - k))) == 0)
				{

					// Header part 4 custom 2 bits

					temp_x = (k >= 1 ? 1 : 0);
					temp_x *= 2;
					temp_y = (k != 1 ? 1 : 0);
					temp_y *= 2;
					bool jump_flag = false;
					if (!compression_data->type_of_compression) // integer compression
					{
						if( ((k == 0) && (BITHL2 >= bit_plane_number )) || ((k == 1) && (BITLH2 >= bit_plane_number)) || ((k == 2) && (BITHH2 >= bit_plane_number)) )
						{
							jump_flag = true;
						}
					}
					if (!jump_flag)
					{
						for (unsigned int i = temp_x ; i < temp_x + 2; ++i)
						{
							for (unsigned int j = temp_y; j < temp_y + 2; ++j)
							{
								
								if((bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
								{
									block_data[block_num].tran_d += (1 << (2 - k));
									block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_D;
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									block_data[block_num].symbol_block[symbol].symbol_val ++;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 4-1\n");
									// finish loops
									i = temp_x + 2;
									j = temp_y + 2;
									break_flag = true;

								}
							}
						}
					}
					if (!break_flag)
					{

						if (!compression_data->type_of_compression) // integer compression
						{
							if( ((k == 0) && (BITHL1 >= bit_plane_number )) || ((k == 1) && (BITLH1 >= bit_plane_number)) || ((k == 2) && (BITHH1 >= bit_plane_number)) )
							{
								continue;
							}
						}
						temp_x = (k >= 1 ? 1 : 0);
						temp_x *= 4;
						temp_y = (k != 1 ? 1 : 0);
						temp_y *= 4;

						for (unsigned int i = temp_x; i < temp_x + 4; ++i )
						{
							for (unsigned int j = temp_y; j < temp_y + 4; ++j)
							{
								
								if ((bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
								{
									block_data[block_num].tran_d += (1 << (2 - k));
									block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_D;
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									block_data[block_num].symbol_block[symbol].symbol_val ++;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 4-2\n");
									// exit the loops
									i = temp_x + 4;
									j = temp_y + 4;
									break_flag = true;
								}
							}
						}
						// bit D is 0
						if((block_data[block_num].tran_d & (1 << (2-k))) == 0)
						{
							block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_TRAN_D;
							block_data[block_num].symbol_block[symbol].symbol_len++;
							block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
							//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
							//printf("SYMBOL 4-3\n");


						}	
					}
				}
			}
		}
	    // if tran_d is 0 error
		if (block_data[block_num].tran_d == 0)
		{
			printf("ERROR: BLOCKSCAN TRAN_D IS 0");
		}

		// now determine TypeCi is needed
		for (unsigned int k = 0; k < 3; ++k)
		{
			if ((block_data[block_num].tran_d & (1 << (2-k))) != 0)
			{
				// Header part 4 custom 2 bits
				if (!compression_data->type_of_compression) // integer compression
				{
					if( ((k == 0) && (BITHL2 >= bit_plane_number )) || ((k == 1) && (BITLH2 >= bit_plane_number)) || ((k == 2) && (BITHH2 >= bit_plane_number)) )
					{
						continue;
					}
				}
				temp_x = (k >= 1 ? 1 : 0);
				temp_x *= 2;
				temp_y = (k != 1 ? 1 : 0);
				temp_y *= 2;
				int p = 1;
				for (unsigned int i = 0; i < 4; ++i)
				{
					if ((block_data[block_num].type_ci[k] << (1 << (3 -i))) != 1 )
					{
						p =0;
						break;
					}
				}
				// p = 0 case
				if ( p == 0)
				{
					unsigned int counter = 0;
					if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
					{
						symbol++;
					}
					block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_CI;
					for (unsigned int i = temp_x ; i < temp_x + 2; ++i)
					{
						for (unsigned int j = temp_y; j < temp_y + 2; ++j)
						{
							if(((block_data[block_num].type_ci[k] & (1 << 3 - counter))) == 0)
							{
								if((bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
								{
									block_data[block_num].type_ci[k] += (1 << (3 - counter));
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									block_data[block_num].symbol_block[symbol].symbol_val ++;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 5-1\n");
									block_data[block_num].symbol_block[symbol].sign <<= 1;
									block_data[block_num].symbol_block[symbol].sign += SIGN(block_string[final_position][(i * BLOCKSIZEIMAGE) + j]);
								}
								else
								{
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 5-2\n");
								}
							}
							else
							{
								temp_value = ((ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j]) & bit_set_plane) > 0 ? 1: 0);
								
								if (!compression_data->type_of_compression) // integer compression
								{
									// header part 4 custom 2 bits TODO CHECK

									if( ((k == 0) && (BITHL2 < bit_plane_number )) || ((k == 1) && (BITLH2 < bit_plane_number)) || ((k == 2) && (BITHH2 < bit_plane_number)) )
									{
										block_data[block_num].children_ref_symbol <<= 1;
										block_data[block_num].children_ref_symbol += temp_value;
										block_data[block_num].children_sym_len ++;
									}
									
								}
								else
								{
									// float compression
									block_data[block_num].children_ref_symbol <<= 1;
									block_data[block_num].children_ref_symbol += temp_value;
									block_data[block_num].children_sym_len ++;
								}
							}
							counter ++;
						}
					}
				}
				else
				{
					// p!= 0 refinement bits
					for (unsigned int i = temp_x ; i < temp_x + 2; ++i)
					{
						for (unsigned int j = temp_y; j < temp_y + 2; ++j)
						{
							temp_value = ((ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j]) & bit_set_plane) > 0 ? 1: 0);
							
							if (!compression_data->type_of_compression) // integer compression
							{
								// header part 4 custom 2 bits TODO CHECK

								if( ((k == 0) && (BITHL2 < bit_plane_number )) || ((k == 1) && (BITLH2 < bit_plane_number)) || ((k == 2) && (BITHH2 < bit_plane_number)) )
								{
									block_data[block_num].children_ref_symbol <<= 1;
									block_data[block_num].children_ref_symbol += temp_value;
									block_data[block_num].children_sym_len ++;
								}
								
							}
							else
							{
								// float compression
								block_data[block_num].children_ref_symbol <<= 1;
								block_data[block_num].children_ref_symbol += temp_value;
								block_data[block_num].children_sym_len ++;
							}
						}
					}
				}			
			}
		}

		// now determine TranGi for the grand children
		if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
		{
			symbol++;
		}
		for (int k = 0; k < 3; ++k)
		{
			// part 4 custom 2 bits
			if (!compression_data->type_of_compression) // integer compression
			{
				if( ((k == 0) && (BITHL1 >= bit_plane_number )) || ((k == 1) && (BITLH1 >= bit_plane_number)) || ((k == 2) && (BITHH1 >= bit_plane_number)) )
				{
					continue;
				}
			}

			//printf ("trand %d: tran GI %d\n", block_data[block_num].tran_d , block_data[block_num].tran_gi);
			if (((block_data[block_num].tran_d & (1 << 2 - k)) != 0) && ((block_data[block_num].tran_gi & (1 << 2 - k)) == 0))
			{
				block_data[block_num].symbol_block[symbol].type = ENUM_TRAN_GI;
				temp_x = (k >= 1 ? 1 : 0);
				temp_x *= 4;
				temp_y = (k != 1 ? 1 : 0);
 				temp_y *= 4;

				bool break_flag = false;
				for (unsigned int i = temp_x; i < temp_x + 4; ++i )
				{
					for (unsigned int j = temp_y; j < temp_y + 4; ++j)
					{
						if( (bit_set_plane & ABSOLUTE(block_string[final_position][(i * BLOCKSIZEIMAGE) + j])) > 0)
						{
							block_data[block_num].symbol_block[symbol].symbol_len++;
							block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
							block_data[block_num].symbol_block[symbol].symbol_val++;
							//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
							//printf("SYMBOL 6-1\n");
							block_data[block_num].symbol_block[symbol].sign <<= 1;
							block_data[block_num].tran_gi += (1 << 2 - k);
							// finish loop
							i = temp_x + 4;
							j = temp_y + 4;
							break_flag = true;
						}
					}
				}
				if (!break_flag)
				{
					block_data[block_num].symbol_block[symbol].symbol_len++;
					block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
					//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
					//printf("SYMBOL 6-2\n");
				}

			}
		}

		// now determine TranHi for the grand children
		if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
		{
			symbol++;
		}
		for (unsigned int i = 0; i < 3; i++)
		{
			// part 4 custom 2 bits integer
			if (!compression_data->type_of_compression) // integer compression
			{
				if( ((i == 0) && (BITHL1 >= bit_plane_number )) || ((i == 1) && (BITLH1 >= bit_plane_number)) || ((i == 2) && (BITHH1 >= bit_plane_number)) )
				{
					continue;
				}
			}
			if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
			{
				symbol++;
			}
			for (unsigned int j = 0; j < 4; j++)
			{
				temp_x = (i >= 1 ? 1 : 0) * 4 + (j >= 2 ? 1 : 0) * 2;
				temp_y = (i != 1 ? 1 : 0) * 4 + (j % 2) * 2;
				if (((block_data[block_num].tran_gi & (1 << (2 -i))) != 0) && ((block_data[block_num].tran_hi[i] & (1 << (3 -j))) == 0 ))
				{
					block_data[block_num].symbol_block[symbol].type = ENUM_TRAN_HI;
					bool break_flag = false;
					for (unsigned int k = temp_x; k < temp_x +2; k++)
					{
						for (unsigned int p = temp_y; p < temp_y + 2; p++)
						{
							if ((bit_set_plane & ABSOLUTE(block_string[final_position][(k * BLOCKSIZEIMAGE) + p])) > 0)
							{
								block_data[block_num].symbol_block[symbol].symbol_len++;
								block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
								block_data[block_num].symbol_block[symbol].symbol_val++;
								block_data[block_num].tran_hi[i] += (1 << (3 -j));
								//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
								//printf("SYMBOL 7-1\n");
								// finish loop
								k = temp_x + 2;
								p = temp_y + 2;
								break_flag = true;
							}
						}
					}
					if (!break_flag)
					{
						block_data[block_num].symbol_block[symbol].symbol_len++;
						block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
						//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
						//printf("SYMBOL 7-2\n");
					}
				}
			}
		}

		for (unsigned i = 0; i < 3; i++)
		{
			// part 4 custom 2 bits integer
			if (!compression_data->type_of_compression) // integer compression
			{
				if( ((i == 0) && (BITHL1 >= bit_plane_number )) || ((i == 1) && (BITLH1 >= bit_plane_number)) || ((i == 2) && (BITHH1 >= bit_plane_number)) )
				{
					continue;
				}
			}
			for (unsigned j = 0; j < 4; j++)
			{
				temp_x = (i >= 1 ? 1 : 0) * 4 + (j >= 2 ? 1 : 0) * 2;
				temp_y = (i != 1 ? 1 : 0) * 4 + (j % 2) * 2;
				
				if ((block_data[block_num].tran_hi[i] & ( 1 << (3 -j ))) != 0)
				{
					short t = 0;
					unsigned int counter = 0;

					for (unsigned int k = 0; k < 4; ++k)
					{
						if((block_data[block_num].type_hi[i*SIZE_TYPE+j] & (1 << (3 - k))) == 0)
						{
							counter++;
						}
					}

					if (counter == 0)
					{
						for (unsigned int k = temp_x; k < temp_x +2; k++)
						{
							for (unsigned int p = temp_y; p < temp_y + 2; p++)
							{
								temp_value = ((ABSOLUTE(block_string[final_position][(k * BLOCKSIZEIMAGE) + p]) & bit_set_plane) > 0 ? 1 : 0);
								
								if (!compression_data->type_of_compression) // integer compression
								{
									// header part 4 custom 2 bits TODO CHECK

									if( ((i == 0) && (BITHL1 < bit_plane_number )) || ((i == 1) && (BITLH1 < bit_plane_number)) || ((i == 2) && (BITHH1 < bit_plane_number)) )
									{
										block_data[block_num].grand_children_ref_symbol[i] <<= 1;
										block_data[block_num].grand_children_ref_symbol[i] += temp_value;
										block_data[block_num].grand_children_sym_len[i]++;
									}
								
								}
								
								else
								{
									block_data[block_num].grand_children_ref_symbol[i] <<= 1;
									block_data[block_num].grand_children_ref_symbol[i] += temp_value;
									block_data[block_num].grand_children_sym_len[i]++;
								}
							}

						}
						continue;
					}

					// if TranHI == 1, then four grand children TypeHij will be scanned

					if (block_data[block_num].symbol_block[symbol].symbol_len != 0)
					{
						symbol++;
					}
					block_data[block_num].symbol_block[symbol].type = ENUM_TYPE_HIJ;

					t = 0;
					for (unsigned int k = temp_x; k < temp_x +2; k++)
					{
						for (unsigned int p = temp_y; p < temp_y + 2; p++)
						{
							//printf("TYPE_HI %d %d %d\n", i, j, block_data[block_num].type_hi[i*SIZE_TYPE+j]);
							if((block_data[block_num].type_hi[i*SIZE_TYPE+j] & (1 << (3 - t))) == 0)
							{
								//printf("TYPE 8 %d : %d\n", bit_set_plane, ABSOLUTE(block_string[final_position][(k * BLOCKSIZEIMAGE) + p]));
								if((bit_set_plane & ABSOLUTE(block_string[final_position][(k * BLOCKSIZEIMAGE) + p])) > 0)
								{	
									block_data[block_num].type_hi[i*SIZE_TYPE+j] += (1 << (3 - t));
									//printf("TYPE_HI SET %d\n", block_data[block_num].type_hi[i*SIZE_TYPE+j]);
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									block_data[block_num].symbol_block[symbol].symbol_val++;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 8-1\n");
									block_data[block_num].symbol_block[symbol].sign <<= 1;
									block_data[block_num].symbol_block[symbol].sign += SIGN(block_string[final_position][(k * BLOCKSIZEIMAGE) + p]);
								}
								else
								{
									block_data[block_num].symbol_block[symbol].symbol_len++;
									block_data[block_num].symbol_block[symbol].symbol_val <<= 1;
									//printf("Symbol val %d: %d\n", symbol, block_data[block_num].symbol_block[symbol].symbol_val);
									//printf("SYMBOL 8-2\n");
								}
							}
							else
							{
								// refinement
								temp_value = ((ABSOLUTE(block_string[final_position][(k * BLOCKSIZEIMAGE) + p]) & bit_set_plane) > 0 ? 1 : 0);
								
								if (!compression_data->type_of_compression) // integer compression
								{
									// header part 4 custom 2 bits TODO CHECK

									if( ((i == 0) && (BITHL1 < bit_plane_number )) || ((i == 1) && (BITLH1 < bit_plane_number)) || ((i == 2) && (BITHH1 < bit_plane_number)) )
									{
									block_data[block_num].grand_children_ref_symbol[i] <<= 1;
									block_data[block_num].grand_children_ref_symbol[i] += temp_value;
									block_data[block_num].grand_children_sym_len[i]++;
									}
								
								}
								else
								{
									block_data[block_num].grand_children_ref_symbol[i] <<= 1;
									block_data[block_num].grand_children_ref_symbol[i] += temp_value;
									block_data[block_num].grand_children_sym_len[i]++;
								}
							}
							t++;
						}
					}
				}
			}
		}

	}
}


void stages_encoding(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number
	)
{
	unsigned int last_block_gaggles = compression_data->segment_size % GAGGLE_SIZE;
	unsigned int total_gaggles = compression_data->segment_size / GAGGLE_SIZE;

	// check fo last gaggle
	if (last_block_gaggles != 0)
	{
		total_gaggles++;
	}
	unsigned char **code_option_gaggles= (unsigned char **)calloc(total_gaggles, sizeof(unsigned char *));
	bool **hit_flag = (bool **)calloc(total_gaggles, sizeof(bool *));

	// 4.5.3.2

	for( unsigned int gaggle_id = 0; gaggle_id < total_gaggles; gaggle_id++)
	{
		unsigned int block_index = gaggle_id * GAGGLE_SIZE;
		code_option_gaggles[gaggle_id] = (unsigned char *)calloc(3, sizeof(unsigned char));
		hit_flag[gaggle_id] = (bool *)calloc(3, sizeof(bool));
		for(unsigned int i = 0; i < 3; i++)
		{
			code_option_gaggles[gaggle_id][i] = 0;
			hit_flag[gaggle_id][i] = false;
		}
		unsigned int blocks_in_gaggle = (block_index + GAGGLE_SIZE <= compression_data->segment_size ? GAGGLE_SIZE : compression_data->segment_size - block_index);
		coding_options(block_data, block_string, compression_data, header_data, segment_number, bit_plane_number, block_index, blocks_in_gaggle, code_option_gaggles[gaggle_id], hit_flag[gaggle_id]);
		gaggle_encode_1(block_data, block_string, compression_data, header_data, segment_number, bit_plane_number, block_index, blocks_in_gaggle, code_option_gaggles[gaggle_id], hit_flag[gaggle_id]);
	}
	for( unsigned int gaggle_id = 0; gaggle_id < total_gaggles; gaggle_id++)
	{
		unsigned int block_index = gaggle_id * GAGGLE_SIZE;
		unsigned int blocks_in_gaggle = (block_index + GAGGLE_SIZE <= compression_data->segment_size ? GAGGLE_SIZE : compression_data->segment_size - block_index);
		gaggle_encode_2(block_data, block_string, compression_data, header_data, segment_number, bit_plane_number, block_index, blocks_in_gaggle, code_option_gaggles[gaggle_id], hit_flag[gaggle_id]);
	}
	for( unsigned int gaggle_id = 0; gaggle_id < total_gaggles; gaggle_id++)
	{
		unsigned int block_index = gaggle_id * GAGGLE_SIZE;
		unsigned int blocks_in_gaggle = (block_index + GAGGLE_SIZE <= compression_data->segment_size ? GAGGLE_SIZE : compression_data->segment_size - block_index);
		gaggle_encode_3(block_data, block_string, compression_data, header_data, segment_number, bit_plane_number, block_index, blocks_in_gaggle, code_option_gaggles[gaggle_id], hit_flag[gaggle_id]);
	}
	ref_bit_end_encode(block_data, block_string, compression_data, header_data, segment_number, bit_plane_number);

}


void pattern_mapping(
	str_symbol_details_t *symbol_details
)
{
	//printf("Symbol length: %d\n", symbol_details->symbol_len);
	switch (symbol_details->symbol_len)
	{
		case 0: return;
		case 1: symbol_details->symbol_mapped_pattern = symbol_details->symbol_val;
				//printf("Symbol mapped pattern: %d\n", symbol_details->symbol_val);
				break;
		case 2: symbol_details->symbol_mapped_pattern = bit2_pattern[symbol_details->symbol_val];
				break;
		case 3:
			
			if (symbol_details->type == ENUM_TYPE_TRAN_D)
			{
				//printf("Symbol type %d %d %d\n", symbol_details->type, bit3_pattern_TranD[symbol_details->symbol_val], symbol_details->symbol_val);
				symbol_details->symbol_mapped_pattern = bit3_pattern_TranD[symbol_details->symbol_val];
			}
			else
			{
				//printf("Symbol type %d %d %d\n", symbol_details->type, bit3_pattern[symbol_details->symbol_val], symbol_details->symbol_val);
				symbol_details->symbol_mapped_pattern = bit3_pattern[symbol_details->symbol_val];
			}
			break;
		case 4: 
			if (symbol_details->type == ENUM_TYPE_CI)
			{
				symbol_details->symbol_mapped_pattern = bit4_pattern_TypeCi[symbol_details->symbol_val];
			}
			else if (symbol_details->type == ENUM_TRAN_HI || symbol_details->type == ENUM_TYPE_HIJ)
			{
				symbol_details->symbol_mapped_pattern = bit4_pattern_TypeHij_TranHi[symbol_details->symbol_val];
			}
			break;
		default:
			//printf("Error: symbol_len not supported\n");
			break;
	}
}

void coding_options(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number,
	unsigned int block_index,
	unsigned int blocks_in_gaggle,
	unsigned char *code_option_gaggle,
	bool *hit_flag
	)
{
	//unsigned int block_sec = 0;
	unsigned int bit_counter_2bits[2] = {0};
	unsigned int bit_counter_3bits[3] = {0};
	unsigned int bit_counter_4bits[4] = {0};

	for (unsigned int block_num = 0; block_num < blocks_in_gaggle; block_num ++)
	{
		unsigned int block_num_index = block_index + block_num; 
		if (block_data[block_num_index].max_ac_bit_size < bit_plane_number)
		{
			continue;
		}
		for( unsigned int symbol_id = 0; symbol_id < MAX_SYMBOLS_IN_BLOCK; symbol_id++)
		{
			// pattern statistics
			//printf("SYMBOL TYPE %d: %d\n",block_data[block_num_index].symbol_block[symbol_id].type, symbol_id);
			if(block_data[block_num_index].symbol_block[symbol_id].type == ENUM_NONE)
			{
				continue;
			}
			else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 1)
			{
				pattern_mapping(&block_data[block_num_index].symbol_block[symbol_id]);
				continue;
			}
			pattern_mapping(&block_data[block_num_index].symbol_block[symbol_id]);
			// entropy coding
			if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 2)
			{
				switch (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern)
				{
					case 0: bit_counter_2bits[0]++; break;
					case 1: bit_counter_2bits[0] += 2; break;
					case 2: bit_counter_2bits[0] += 3; break;
					case 3: bit_counter_2bits[0] += 3; break;
					default: printf("Error: symbol_mapped_pattern not supported\n"); break;
				}
				bit_counter_2bits[1] += 2; // uncoded
			}
			else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 3)
			{
				// try option 0
				if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 2)
				{
					bit_counter_3bits[0] += block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern + 1;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 5)
				{
					bit_counter_3bits[0] += 5;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 7)
				{
					bit_counter_3bits[0] += 6;
				}
				else
				{
					printf("Error: symbol_mapped_pattern not supported\n");
				}
				// option 1
				if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 1)
				{
					bit_counter_3bits[1] += 2;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 3)
				{
					bit_counter_3bits[1] += 3;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 7)
				{
					bit_counter_3bits[1] += 4;
				}
				else
				{
					printf("Error: symbol_mapped_pattern not supported\n");
				}
				// uncoded
				bit_counter_3bits[2] += 3;
			}
			else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 4)
			{
				// ty 0
				if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 3)
				{
					bit_counter_4bits[0] += block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern + 1;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 7)
				{
					bit_counter_4bits[0] += 7;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 15)
				{
					bit_counter_4bits[0] += 8;
				}
				else
				{
					printf("Error: symbol_mapped_pattern not supported\n");
				}
				// try 1

				if (block_data[block_num_index].symbol_block[symbol_id]. symbol_mapped_pattern <= 1)
				{
					bit_counter_4bits[1] += 2;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 3)
				{
					bit_counter_4bits[1] += 3;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 5)
				{
					bit_counter_4bits[1] += 4;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 11)
				{
					bit_counter_4bits[1] += 6;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 15)
				{
					bit_counter_4bits[1] += 7;
				}
				else
				{
					printf("Error: symbol_mapped_pattern not supported\n");
				}

				// try 2
				if (block_data[block_num_index].symbol_block[symbol_id]. symbol_mapped_pattern <= 3)
				{
					bit_counter_4bits[2] += 3;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 7)
				{
					bit_counter_4bits[2] += 4;
				}
				else if (block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern <= 15)
				{
					bit_counter_4bits[2] += 5;
				}
				else
				{
					printf("Error: symbol_mapped_pattern not supported\n");
				}
				// tyr 3
				bit_counter_4bits[3] += 4;
			}	

		}
	}
	// determine code ID

	// 2 bit
	if(	bit_counter_2bits[0] < bit_counter_2bits[1] )
	{
		code_option_gaggle[0] = 0; // codes. 
	}		
	else
	{
		code_option_gaggle[0] = 1; // no-coding
	}

	// 3 bit
	if((bit_counter_3bits[2] <= bit_counter_3bits[0]) && (bit_counter_3bits[2] <= bit_counter_3bits[1]))
	{
		code_option_gaggle[1] = 3;
	}
	else if((bit_counter_3bits[0] <= bit_counter_3bits[1]) &&  (bit_counter_3bits[0] <= bit_counter_3bits[2]))
	{
		code_option_gaggle[1] = 0;
	}
	else if((bit_counter_3bits[1]  <= bit_counter_3bits[0]) && (bit_counter_3bits[1]  <= bit_counter_3bits[2]))
	{
		code_option_gaggle[1] = 1;
	}
	// 4-bit codeword
	if((bit_counter_4bits[3]  <= bit_counter_4bits[1]) && (bit_counter_4bits[3]  <= bit_counter_4bits[0])&& (bit_counter_4bits[3]  <= bit_counter_4bits[2]))
	{
		code_option_gaggle[2] = 3;	
	}
	else if((bit_counter_4bits[0]  <= bit_counter_4bits[1]) && (bit_counter_4bits[0]  <= bit_counter_4bits[2]) && (bit_counter_4bits[0]  <= bit_counter_4bits[3]))
	{
		code_option_gaggle[2] = 0;
	}
	else if((bit_counter_4bits[1]  <= bit_counter_4bits[0]) && (bit_counter_4bits[1]  <= bit_counter_4bits[2])&& (bit_counter_4bits[1]  <= bit_counter_4bits[3]))
	{
		code_option_gaggle[2] = 1;
	}		
	else if((bit_counter_4bits[2]  <= bit_counter_4bits[1]) && (bit_counter_4bits[2]  <= bit_counter_4bits[0])&& (bit_counter_4bits[2]  <= bit_counter_4bits[3]))
	{
		code_option_gaggle[2] = 2;	
	}					

}

void rice_coding(
	compression_image_data_t *compression_data,
	unsigned int segment_num,
	short input_value,
	short bit_length,
	unsigned char *code_option
)
{
	//printf("Bit length: %d\n", bit_length);
	switch (bit_length)
	{
	case 0: // no need to process
		return;
		break;
	case 1:
		write_to_the_output_segment(compression_data->segment_list,input_value,1,segment_num);
		break;
	case 2:
		// output FS code.
		if (code_option[0] == 1)
		{
			write_to_the_output_segment(compression_data->segment_list,input_value,2,segment_num);
		}
		else if (code_option[0] == 0)
		{
			if (input_value <= 2)
			{
				write_to_the_output_segment(compression_data->segment_list,0,input_value,segment_num);
				write_to_the_output_segment(compression_data->segment_list,1,1,segment_num);
			}
			else
			{
				write_to_the_output_segment(compression_data->segment_list,0,3,segment_num);
			}
		}
		else
		{
			//printf("Error: rice coding code_option not supported\n");
		}
		break;
	case 3:
		//printf("InputVal: %d %d\n", input_value, code_option[1]);
		if (code_option[1] == 0)
		{
			if (input_value <= 2)
			{
				write_to_the_output_segment(compression_data->segment_list,0,input_value,segment_num);
				write_to_the_output_segment(compression_data->segment_list,1,1,segment_num);
			}
			else if (input_value <= 5)
			{
				write_to_the_output_segment(compression_data->segment_list,0,3,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value - 3,2,segment_num);
			}
			else if (input_value <= 7)
			{
				write_to_the_output_segment(compression_data->segment_list,0,3,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value,3,segment_num);
			}
			else
			{
				printf("Error: rice coding code_option not supported\n");
			}
		}
		else if (code_option[1] == 1)
		{
			if (input_value <= 1)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value + 2,2,segment_num);
			}
			else if (input_value <= 3)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value,3, segment_num);
			}
			else if (input_value <= 7)
			{
				write_to_the_output_segment(compression_data->segment_list,0,2,segment_num);
				switch (input_value)

				{
				case 4:
					write_to_the_output_segment(compression_data->segment_list,2,2,segment_num);
					break;
				case 5:
					write_to_the_output_segment(compression_data->segment_list,3,2,segment_num);
					break;
				case 6:
					write_to_the_output_segment(compression_data->segment_list,0,2,segment_num);
					break;
				case 7:
					write_to_the_output_segment(compression_data->segment_list,1,2,segment_num);
					break;
				default:
					printf("Error: rice coding value not supported\n");
					break;
				}

			}
			else
			{
				printf("Error: rice coding code_option not supported\n");
			}
		}
		else if (code_option[1] == 3) // uncoded
		{
			write_to_the_output_segment(compression_data->segment_list,input_value,3,segment_num);
		}
		break;
	case 4: // 4 bits
		//printf("InputVal: %d %d\n", input_value, code_option[2]);
		if (code_option[2] == 0)
		{
			if (input_value <= 3)
			{
				write_to_the_output_segment(compression_data->segment_list,0,input_value,segment_num);
				write_to_the_output_segment(compression_data->segment_list,1,1,segment_num);
			}
			else if (input_value <= 7)
			{
				write_to_the_output_segment(compression_data->segment_list,0,5,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value - 4,2,segment_num);
			}
			else if (input_value <= 15)
			{
				write_to_the_output_segment(compression_data->segment_list,0,4,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value, 4,segment_num);
			}
			else
			{
				printf("Error: rice coding input value not supported\n");
			}
		}
		else if (code_option[2] == 1)
		{
			if (input_value <= 1)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value + 2 ,2,segment_num);
			}
			else if (input_value <= 3)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value,3,segment_num);
			}
			else if (input_value <= 5)
			{
				write_to_the_output_segment(compression_data->segment_list,0,2,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value - 2,2,segment_num);
			}
			else if (input_value <= 11)
			{
				write_to_the_output_segment(compression_data->segment_list,0,3,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value - 6,3,segment_num);
			}
			else if (input_value <= 15)
			{
				write_to_the_output_segment(compression_data->segment_list,0,3,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value, 4,segment_num);
			}
			else
			{
				printf("Error: rice coding input value not supported\n");
			}
		}
		else if (code_option[2] == 2 ) // two bits spliting
		{
			if (input_value <= 3)
			{
				write_to_the_output_segment(compression_data->segment_list, input_value + 4 , 3,segment_num);
			}
			else if (input_value <= 7)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value, 4,segment_num);
			}
			else if (input_value <= 11)
			{
				write_to_the_output_segment(compression_data->segment_list,0,2,segment_num);
				write_to_the_output_segment(compression_data->segment_list,input_value - 4, 3,segment_num);
			}
			else if (input_value <= 15)
			{
				write_to_the_output_segment(compression_data->segment_list,input_value -12, 5,segment_num);
			}
			else
			{
				printf("Error: rice coding input value not supported\n");
			}
		}
		else if (code_option[2] == 3) // uncoded
		{
			write_to_the_output_segment(compression_data->segment_list,input_value,4,segment_num);
		}
		else
		{
			printf("Error: rice coding code_option not supported\n");
		}
		break;
	default:
		printf("Error: rice coding bit_length not supported\n");
		break;
	}
}


void gaggle_encode_1(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number,
	unsigned int block_index,
	unsigned int blocks_in_gaggle,
	unsigned char *code_option_gaggle,
	bool *hit_flag
	)
{
	for (unsigned int block_num = 0; block_num < blocks_in_gaggle; block_num ++)
	{
		// get the block index
		unsigned int block_num_index = block_index + block_num; 
		if (block_data[block_num_index].max_ac_bit_size < bit_plane_number)
		{
			continue;
		}
		for( unsigned int symbol_id = 0; symbol_id < MAX_SYMBOLS_IN_BLOCK; symbol_id++)
		{
			if (block_data[block_num_index].symbol_block[symbol_id].type == ENUM_TYPE_P)
			{
				//printf("SYMBOL LENGTH: %d\n", block_data[block_num_index].symbol_block[symbol_id].symbol_len);
				switch (block_data[block_num_index].symbol_block[symbol_id].symbol_len)
				{
				case 1:
				case 2:
				case 3:
				{
					if (block_data[block_num_index].symbol_block[symbol_id].symbol_len > 1)
					{
						if (hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] == false)
						{
							hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] = true;
							
							//printf("BLOCK OF 3\n");
							if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 2)
							{
								
								write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[0],1,segment_number);
							}
							else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 3)
							{
								write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[1],2,segment_number);
							}
							else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 4)
							{
								write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[2],2,segment_number);
							}
							else
							{
								printf("Error: symbol_len not supported\n");
							}
						}
					}
					//printf("RICE CODING START 1\n");
					//printf("input value: %d symbol_id %d\n",block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern, symbol_id);
					rice_coding(compression_data,segment_number ,block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern, block_data[block_num_index].symbol_block[symbol_id].symbol_len, code_option_gaggle);
					//printf("RICE CODING END 1\n");
					unsigned int counter = 0;

					for (unsigned int i = 0; i < block_data[block_num_index].symbol_block[symbol_id].symbol_len; ++i)
					{
						if ((block_data[block_num_index].symbol_block[symbol_id].symbol_val & (1 << i)) > 0)
						{
							counter++;
						}
					}
					write_to_the_output_segment(compression_data->segment_list,block_data[block_num_index].symbol_block[symbol_id].sign,counter,segment_number);
					// reset symbol values
					block_data[block_num_index].symbol_block[symbol_id].symbol_val = 0;
					block_data[block_num_index].symbol_block[symbol_id].symbol_len = 0;
					block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern = 0;
					block_data[block_num_index].symbol_block[symbol_id].sign = 0; 
					block_data[block_num_index].symbol_block[symbol_id].type = 0;
					//printf("TYPE POST RESTET %d\n", block_data[block_num_index].symbol_block[symbol_id].type );
					break;
				}
				default:
					printf("Error: symbol_len not supported\n");
					break;
				}
			}
		}
	}
}

void gaggle_encode_2(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number,
	unsigned int block_index,
	unsigned int blocks_in_gaggle,
	unsigned char *code_option_gaggle,
	bool *hit_flag
	)
{

	for (unsigned int block_num = 0; block_num < blocks_in_gaggle; block_num ++)
	{
		unsigned int block_num_index = block_index + block_num; 
		if (block_data[block_num_index].max_ac_bit_size < bit_plane_number)
		{
			continue;
		}
		for (unsigned int symbol_id = 0; symbol_id < MAX_SYMBOLS_IN_BLOCK; symbol_id++)
		{
			switch (block_data[block_num_index].symbol_block[symbol_id].type)
			{
			case ENUM_TYPE_TRAN_B:
			case ENUM_TYPE_TRAN_D:
			case ENUM_TYPE_CI:
		        if (block_data[block_num_index].symbol_block[symbol_id].symbol_len > 1)
		        {
		            if (hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] == false)
		            {
		                hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] = true;
		                
		                if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 2)
		                {
		                    write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[0],1,segment_number);
		                }
		                else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 3)
		                {
		                    write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[1],2,segment_number);
		                }
		                else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 4)
		                {
		                   write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[2],2,segment_number);
		                }
		                else
		                {
		                    printf("Error: symbol_len not supported\n");
		                }
		            }
		        }
				//printf("RICE CODING START 2\n");
				rice_coding(compression_data,segment_number,block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern, block_data[block_num_index].symbol_block[symbol_id].symbol_len, code_option_gaggle);
				//printf("RICE CODING END 2\n");
				if (block_data[block_num_index].symbol_block[symbol_id].type == ENUM_TYPE_CI)
				{
					unsigned int counter = 0;
					for (unsigned int i = 0; i < block_data[block_num_index].symbol_block[symbol_id].symbol_len; ++i)
					{
						if ((block_data[block_num_index].symbol_block[symbol_id].symbol_val & (1 << i)) > 0)
						{
							counter++;
						}
						
					}
					write_to_the_output_segment(compression_data->segment_list,block_data[block_num_index].symbol_block[symbol_id].sign,counter,segment_number);
				}
				block_data[block_num_index].symbol_block[symbol_id].symbol_val = 0;
				block_data[block_num_index].symbol_block[symbol_id].symbol_len = 0;
				block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern = 0;
				block_data[block_num_index].symbol_block[symbol_id].sign = 0; 
				block_data[block_num_index].symbol_block[symbol_id].type = 0;
				break;
			
			default:
				break;
			}
		}
	}

}


void gaggle_encode_3(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number,
	unsigned int block_index,
	unsigned int blocks_in_gaggle,
	unsigned char *code_option_gaggle,
	bool *hit_flag
	)
{
	for (unsigned int block_num = 0; block_num < blocks_in_gaggle; block_num ++)
	{
		unsigned int block_num_index = block_index + block_num; 
		if (block_data[block_num_index].max_ac_bit_size < bit_plane_number)
		{
			continue;
		}
		for (unsigned int symbol_id = 0; symbol_id < MAX_SYMBOLS_IN_BLOCK; symbol_id++)
		{
			switch (block_data[block_num_index].symbol_block[symbol_id].type)
			{
			case ENUM_TRAN_GI:
			case ENUM_TRAN_HI:
			case ENUM_TYPE_HIJ:
				if (block_data[block_num_index].symbol_block[symbol_id].symbol_len > 1)
				{
					if (hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] == false)
					{
						hit_flag[block_data[block_num_index].symbol_block[symbol_id].symbol_len - 2] = true;
						if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 2)
						{
							write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[0],1,segment_number);
						}
						else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 3)
						{
							write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[1],2,segment_number);
						}
						else if (block_data[block_num_index].symbol_block[symbol_id].symbol_len == 4)
						{
							write_to_the_output_segment(compression_data->segment_list,code_option_gaggle[2],2,segment_number);
						}
						else
						{
							printf("Error: symbol_len not supported\n");
						}
					}
				}
				//printf("RICE CODING START 3\n");
				rice_coding(compression_data,segment_number,block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern, block_data[block_num_index].symbol_block[symbol_id].symbol_len, code_option_gaggle);
				//printf("RICE CODING END 3\n");
				if (block_data[block_num_index].symbol_block[symbol_id].type == ENUM_TYPE_HIJ)
				{
					unsigned int counter = 0;
					for (unsigned int i = 0; i < block_data[block_num_index].symbol_block[symbol_id].symbol_len; ++i)
					{
						if ((block_data[block_num_index].symbol_block[symbol_id].symbol_val & (1 << i)) > 0)
						{
							counter++;
						}
					}
					write_to_the_output_segment(compression_data->segment_list,block_data[block_num_index].symbol_block[symbol_id].sign,counter,segment_number);
				}
				block_data[block_num_index].symbol_block[symbol_id].symbol_val = 0;
				block_data[block_num_index].symbol_block[symbol_id].symbol_len = 0;
				block_data[block_num_index].symbol_block[symbol_id].symbol_mapped_pattern = 0;
				block_data[block_num_index].symbol_block[symbol_id].sign = 0;
				block_data[block_num_index].symbol_block[symbol_id].type = 0;
			break;
			}
		}
	}
}

void ref_bit_end_encode(	
	block_data_t *block_data,
	int **block_string,
	compression_image_data_t *compression_data,
	header_data_t *header_data,
	unsigned int segment_number,
	unsigned int bit_plane_number
	)
{
	for (unsigned int block_num = 0; block_num < compression_data->segment_size; ++block_num)
	{	
		//printf("PARENT SYM LENG %d\n", block_data[block_num].parent_sym_len);
		if (block_data[block_num].parent_sym_len > 0)
		{
			write_to_the_output_segment(compression_data->segment_list,
			block_data[block_num].parent_ref_symbol,
			block_data[block_num].parent_sym_len,
			segment_number);

			// reste to zero
			block_data[block_num].parent_ref_symbol = 0;
			block_data[block_num].parent_sym_len = 0;
		}

		//printf("CHILDREN SYM LENG %d\n", block_data[block_num].children_sym_len);
		if (block_data[block_num].children_sym_len > 0)
		{
			write_to_the_output_segment(compression_data->segment_list,
			block_data[block_num].children_ref_symbol,
			block_data[block_num].children_sym_len,
			segment_number);

			// reste to zero
			block_data[block_num].children_ref_symbol = 0;
			block_data[block_num].children_sym_len = 0;
		}

		for (unsigned int i = 0; i < 3; ++i)
		{
			//printf("GRANCHILDREN SYM LENG %d %d\n",i, block_data[block_num].grand_children_sym_len[i]);
			if (block_data[block_num].grand_children_sym_len[i] > 0)
			{
				write_to_the_output_segment(compression_data->segment_list,
				block_data[block_num].grand_children_ref_symbol[i],
				block_data[block_num].grand_children_sym_len[i],
				segment_number);

				// reste to zero
			}
			block_data[block_num].grand_children_sym_len[i] = 0;
			block_data[block_num].grand_children_ref_symbol[i] = 0;
		}
	} 

}