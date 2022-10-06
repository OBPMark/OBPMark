#include "../lib_functions.h"

// auxiliar fuctions
void ccsds_wavelet_transform_1D(const int* A, int* B, const int size);
void ccsds_wavelet_transform_1D(const float* A, float* B, const int size);
void ccsds_wavelet_transform_2D (int** A, const int w_size, const int h_size);
void ccsds_wavelet_transform_2D (float** A, const int w_size, const int h_size);
int** readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns);
void  writeBMP(char* filename, int** output ,unsigned int w_size, unsigned int h_size);
void  writeBMP(char* filename, float** output ,unsigned int w_size, unsigned int h_size);
void coeff_regroup(int **transformed_image, unsigned int h_size, unsigned int w_size);
void build_block_string(int **transformed_image, unsigned int h_size, unsigned int w_size, long **block_string);
void header_inilization(header_struct *header_pointer, BOOL DWT_type, unsigned int pad_rows, unsigned int image_width, BOOL first, BOOL last);
void encode_dc(block_struct* list_of_process_data, long **block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks,unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer);
void dpcm_dc_mapper(unsigned long *shifted_dc, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_entropy_encoder(block_struct* list_of_process_data, unsigned int block_position, unsigned short *dc_remainer, unsigned char quantization_factor, header_struct *header_pointer, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_encoder(unsigned int gaggle_id, unsigned int block_position, block_struct* list_of_process_data, header_struct *header_pointer, unsigned long *dc_mapper, short N, int star_index, int gaggles, int max_k, int id_length);
//void acb_pe_encoding();
void ac_depth_encoder(header_struct *header_pointer, unsigned int blocks_in_segment, unsigned short *bit_max_in_segment,  BOOL *segment_full);
void dpcm_ac_mapper(header_struct *header_pointer, unsigned int blocks_in_segment, unsigned char N, unsigned short *bit_max_in_segment,unsigned short *mapped_ac);
void ac_gaggle_encoding(header_struct *header_pointer, unsigned short *mapped_ac, signed int gaggle_start_index, unsigned char gaggles, unsigned char max_k, unsigned char id_lengh, unsigned char N);
//void block_scan_encode();
unsigned long conversion_twos_complement(long original, short leftmost);
void convert_to_bits_segment(FILE *ofp, block_struct segment_data);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init(DataObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}

void init(DataObject *device_object, int platform ,int device, char* device_name)
{
	// empty
}

bool device_memory_init(DataObject *device_object)
{
	// empty
}

void copy_data_to_cpu(DataObject *device_object, int* image_data)
{
	return;
}

void encode_engine(DataObject *device_object, int* image_data_linear)
{
   
	
	// first read the data and copy to the input image
    unsigned int size = sizeof(float) * device_object->h_size * device_object->w_size;
	// calculate the number of pading
	unsigned int h_size_padded = 0;
	unsigned int w_size_padded = 0;
	unsigned int pad_rows = device_object->pad_rows;
	unsigned int pad_colums = device_object->pad_columns;

	// create te new size
	h_size_padded = device_object->h_size + device_object->pad_rows;
	w_size_padded = device_object->w_size + device_object->pad_columns;
    // read imput image

	int  **image_data = NULL;
	image_data = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		image_data[i] = (int *)calloc(w_size_padded, sizeof(int));
	}
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			image_data[i][j] = image_data_linear[i * h_size_padded + j];
		}
	}

    //int** image_data = readBMP(device_object->filename_input, device_object->pad_rows, device_object->pad_columns);
   
	int  **transformed_image = NULL;
	transformed_image = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		transformed_image[i] = (int *)calloc(w_size_padded, sizeof(int));
	}
	
	// start the 2D DWT operation 
	clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_dwt);

	// pass to the 2D DWT
	if (device_object->type)
	{
		float  **aux_data = NULL;
		aux_data = (float**)calloc(h_size_padded, sizeof(float *));
		for(unsigned i = 0; i < h_size_padded; i++){
			aux_data[i] = (float *)calloc(w_size_padded, sizeof(float));
		}
		// convert the original interger image unsigned int to float
		// convert image to float
		for( int i=0 ; i < device_object->h_size ; i++)
   		{
    		for( int j=0 ; j < device_object->w_size ; j++)
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
    	//writeBMP(device_object->filename_output, transformed_image,*device_object->w_size,*device_object->h_size );
		for(unsigned i = 0; i < h_size_padded; i++){
			free(aux_data[i]);
		}
		free(aux_data);
	}
	else{
		// integer encoding
		unsigned int iteration = 0;
		while(iteration != LEVELS_DWT){
			if (iteration == 0){
				ccsds_wavelet_transform_2D(image_data,w_size_padded, h_size_padded);
			}
			else{
				// create a subimage from the original
				unsigned int new_h_size = device_object->h_size/ (2*iteration);
				unsigned int new_w_size = device_object->w_size/ (2*iteration);
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
    	//writeBMP(device_object->filename_output, aux_data,*device_object->w_size,*device_object->h_size );
	}

	writeBMP(device_object->filename_output, transformed_image, w_size_padded, h_size_padded);

	// end the 2D DWT
	clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_dwt);
	// start of the BPE
	clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_bpe);
	// transformed image haves the procesed output image in integer
	// Coeficients reagroup
	/*
	##########################################################################################################
	# This function take the image that has been processed for each of the levels of the DWT 2D and
	# re-arrange the data so each 8 by 8 block contains a family of de DC componet been the DC componet
	# in 0 0 of that block.
	##########################################################################################################
	*/
	// Step 1 transform the image 
	coeff_regroup(transformed_image, h_size_padded, w_size_padded);
	// Step 2 get the block string data
	
	// build_block_string
	/*
	##########################################################################################################
	# This fuction takes the rearange image and creates total_blocks
	# So a 8 by 8 data is store in block_string[0][0] to block_string[0][63]
	# each position in x of block_string contais 64 data of the image
	##########################################################################################################
	*/
	unsigned int total_blocks =  (h_size_padded / BLOCKSIZEIMAGE )*(w_size_padded/ BLOCKSIZEIMAGE);
	long **block_string = NULL;
	block_string = (long **)calloc(total_blocks,sizeof(long *));
	for(unsigned int i = 0; i < total_blocks ; i++)
	{
		block_string[i] = (long *)calloc(BLOCKSIZEIMAGE * BLOCKSIZEIMAGE,sizeof(long));
	}
	build_block_string(transformed_image, h_size_padded, w_size_padded,block_string);
	// clean image
	for(unsigned int i = 0; i < h_size_padded; i++){
			free(transformed_image[i]);
		}
	free(transformed_image);
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
		header_struct *ptr_header = NULL;
		ptr_header = (header_struct *) calloc( 1,sizeof(header_struct));
		// inicialize header
		
		if (block_counter == 0){
			// first block
			header_inilization(ptr_header, DWT_type, pad_rows, device_object->w_size, TRUE, FALSE);
		}
		else if (block_counter + SEGMENTSIZE == total_blocks)
		{
			// last block
			header_inilization(ptr_header, DWT_type, pad_rows, device_object->w_size, FALSE, TRUE);
		}
		else if (block_counter + SEGMENTSIZE > total_blocks)
		{			
			// Special sitiuation when the number of blocks per segment are not the same
			header_inilization(ptr_header, DWT_type, pad_rows, device_object->w_size, FALSE, TRUE);
			ptr_header->header.part1.part_3_flag = TRUE;
			ptr_header->header.part3.seg_size_blocks_20bits = total_blocks - block_counter;

			ptr_header->header.part1.part_2_flag = TRUE;
			ptr_header->header.part2.seg_byte_limit_27bits = BITSPERFIXEL * (total_blocks - block_counter) * 64/8;
		}
		else
		{
			header_inilization(ptr_header, DWT_type, pad_rows, device_object->w_size, FALSE, FALSE);
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
		// Step 5 encode AC
		// missing IF
		//acb_pe_encoding();
		// copy data
		list_of_process_data[segment].dc_remainer = dc_remainer;
		list_of_process_data[segment].quantization_factor = quantization_factor;
		list_of_process_data[segment].header = (header_struct_base *) calloc( 1,sizeof(header_struct_base));
		list_of_process_data[segment].header->part1 = ptr_header->header.part1;
		list_of_process_data[segment].header->part2 = ptr_header->header.part2;
		list_of_process_data[segment].header->part3 = ptr_header->header.part3;
		list_of_process_data[segment].header->part4 = ptr_header->header.part4;
		++segment;
		free(ptr_header);

	}
	// end processing
	clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_bpe);
	// store the data
	FILE *ofp;

    ofp = fopen(device_object->filename_output, "w");

    if (ofp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n",
             device_object->filename_output);
    exit(1);
    }
	// loop over the data
	for (unsigned int i = 0; i < number_of_segment; ++i)
	{
		convert_to_bits_segment(ofp, list_of_process_data[i]);
	}

	fclose(ofp);
	// print the timing
	

	
	

}
void get_elapsed_time(DataObject *device_object, bool csv_format){
	unsigned long milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0, miliseconds_bpe = 0;
	milliseconds =  (device_object->end_dwt.tv_sec - device_object->start_dwt.tv_sec) * 1000 + (device_object->end_dwt.tv_nsec - device_object->start_dwt.tv_nsec) / 1000000;
	miliseconds_bpe = (device_object->end_bpe.tv_sec - device_object->start_bpe.tv_sec) * 1000 + (device_object->end_bpe.tv_nsec - device_object->start_bpe.tv_nsec) / 1000000;

	if (csv_format){
         printf("%lu;%lu;%lu;%lu;\n", milliseconds_h_d,milliseconds,miliseconds_bpe,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %lu miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel DWT: %lu miliseconds\n", milliseconds);
         printf("Elapsed time kernel BPE: %lu miliseconds\n", miliseconds_bpe);
         printf("Elapsed time Device->Host: %lu miliseconds\n", milliseconds_d_h);
    }
	
}




// Internal Operations

void build_block_string(int **transformed_image, unsigned int h_size, unsigned int w_size, long **block_string)
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

void coeff_regroup(int **transformed_image, unsigned int h_size, unsigned int w_size)
{

	int  **temp = NULL;
	temp = (int**)calloc(h_size, sizeof(int *));
	for(unsigned i = 0; i < h_size; i++){
		temp[i] = (int *)calloc(w_size, sizeof(int));
	}
	// HH1 band. Starts with grandchildren of family 2
	for (unsigned int i = (h_size>>1); i < h_size; i+=4)
	{
		for (unsigned int j = (w_size>>1); j < w_size; j+=4)
		{
			unsigned int x = ((i - (h_size>>1)) << 1);
			unsigned int y = ((j - (w_size>>1)) << 1);
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
	for (unsigned int i = 0; i < (h_size >> 1) ; i+=4)
	{
		for (unsigned int j = (w_size>>1); j < w_size; j+=4)
		{
			unsigned int x = (i  << 1);
			unsigned int y = ((j - (w_size>>1)) << 1);
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
	for (unsigned int i = (h_size>>1); i < h_size; i+=4)
	{
		for (unsigned int j = 0; j < (w_size>>1); j+=4)
		{
			unsigned int x = ((i - (h_size>>1)) << 1);
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
	for (unsigned int i = (h_size>>2); i < (h_size>>1); i+=2)
	{
		for (unsigned int j = (w_size>>2); j < (w_size>>1); j+=2)
		{
			unsigned int x = ((i - (h_size>>2)) <<2);
			unsigned int y = ((j - (w_size>>2)) <<2);
			temp[x + 2][y + 2] =  transformed_image[i][j];			
			temp[x + 2][y + 3] =  transformed_image[i][j + 1];			
			temp[x + 3][y + 2] =  transformed_image[i + 1][j];			
			temp[x + 3][y + 3] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// HL2 band. children of family 0
	for (unsigned int i = 0; i < (h_size>>2); i+=2)
	{
		for (unsigned int j = (w_size>>2); j < (w_size>>1); j+=2)
		{
			unsigned int x = (i <<2);
			unsigned int y = ((j - (w_size>>2)) <<2);
			temp[x][y + 2] =  transformed_image[i][j];			
			temp[x][y + 3] =  transformed_image[i][j + 1];			
			temp[x + 1][y + 2] =  transformed_image[i + 1][j];			
			temp[x + 1][y + 3] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// LH2 band. children of family 1
	for (unsigned int i = (h_size>>2); i < (h_size>>1); i+=2)
	{
		for (unsigned int j = 0; j < (w_size>>2); j+=2)
		{
			unsigned int x = ((i - (h_size>>2)) <<2);
			unsigned int y = (j<<2);
			temp[x + 2][y] =  transformed_image[i][j];			
			temp[x + 2][y + 1] =  transformed_image[i][j + 1];			
			temp[x + 3][y] =  transformed_image[i + 1][j];			
			temp[x + 3][y + 1] =  transformed_image[i + 1][j + 1];
			
		}
	}
	// HH3 band parent family 2
	unsigned int x = (h_size>>3);
	unsigned int y = (w_size>>2);
	for (unsigned int i = (h_size>>3); i < (h_size>>2); ++i)
	{
		for (unsigned int j = (w_size>>3); j < (w_size>>2); ++j)
		{
			temp[((i - x) <<3) + 1][((j - (w_size>>3)) <<3) + 1] = transformed_image[i][j];
		}
	}
	// HL3 band parent family 0
	for (unsigned int i = 0; i < (h_size>>3); ++i)
	{
		for (unsigned int j = (w_size>>3); j < (w_size>>2); ++j)
		{
			temp[i << 3][((j - (w_size>>3)) <<3) + 1] = transformed_image[i][j];
		}
	}
	// LH3 band parent family 1
	for (unsigned int i = (h_size>>3); i < (h_size>>2); ++i)
	{
		for (unsigned int j = 0; j < (w_size>>3); ++j)
		{
			temp[((i - x) <<3) + 1][j<<3] = transformed_image[i][j];
		}
	}
	// LL3 band , DC components
	for (unsigned int i = 0; i < (h_size>>3); ++i)
	{
		for (unsigned int j = 0; j < (w_size>>3); ++j)
		{
			temp[i<<3][j<<3] =  transformed_image[i][j];
		}
	}
	// copy the values to transformed_image
	for(unsigned int i = 0; i < h_size; i++)
	{		
		for(unsigned int j = 0; j < w_size; j++)		
		{
				transformed_image[i][j] = temp[i][j];
		}
	}
	// finish conversion

}


/*int** readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns)
{
	BMP Image;
   	Image.ReadFromFile( filename );
    int size_output =  Image.TellWidth() * Image.TellHeight();
	int  **data_bw = NULL;
	data_bw = (int**)calloc(Image.TellHeight() + pad_rows, sizeof(int *));
	for(unsigned i = 0; i < Image.TellHeight() + pad_rows; i++){
		data_bw[i] = (int *)calloc(Image.TellWidth() + pad_columns, sizeof(int));
	}
   	// convert each pixel to greyscale
   	for( int i=0 ; i < Image.TellHeight() ; i++)
   	{
    	for( int j=0 ; j < Image.TellWidth() ; j++)
    	{
			data_bw[i][j] =  (Image(j,i)->Red + Image(j,i)->Green + Image(j,i)->Blue)/3;
    	}
   }
   //we need to duplicate rows and columns to be in BLOCKSIZEIMAGE
   for(unsigned int i = 0; i < pad_rows ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_columns; j++)
			data_bw[i + Image.TellHeight()][j] = data_bw[Image.TellHeight() - 1][j];
	}

	for(unsigned int i = 0; i < pad_columns ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_rows ; j++)
			data_bw[j][i + Image.TellWidth()] = data_bw[j][Image.TellWidth() - 1];
	}

   return data_bw;
}*/
void  writeBMP(char* filename, int** output ,unsigned int w_size, unsigned int h_size){
    float  **aux_data = NULL;
	aux_data = (float**)calloc(h_size, sizeof(float *));
	for(unsigned i = 0; i < h_size; i++){
		aux_data[i] = (float *)calloc(w_size, sizeof(float));
	}
	for(unsigned int i=0; i<h_size; ++i){
        for(unsigned int j=0; j<w_size; ++j){
            aux_data[i][j] = float(output[i][j]);
        }
    }
    writeBMP(filename, aux_data ,w_size, h_size);
}

void writeBMP(char* filename, float** output ,unsigned int w_size, unsigned int h_size){


    BMP output_image;
    output_image.SetSize(w_size, h_size);
    output_image.SetBitDepth(32);
    for(unsigned int i=0; i<h_size; ++i){
        for(unsigned int j=0; j<w_size; ++j){
            float a = output [i][j];
            output_image(j,i)->Blue=int(a); 
            output_image(j,i)->Red=int(a);
            output_image(j,i)->Green=int(a );
        }
    }
    output_image.WriteToFile(filename);
}

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

// header functions 
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

void encode_dc(block_struct* list_of_process_data, long **block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks, unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer)
{
	long  max_dc_per_segment = 0;
	unsigned long temp_bit_max_dc = 0;

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
		if(TOPOSITIVE(block_string[block_position_in_segment][0]) > TOPOSITIVE(max_dc_per_segment))
		{
			// update the value of max dc in this segment
			max_dc_per_segment = block_string[block_position_in_segment][0];
		}
		
		if((unsigned long)(TOPOSITIVE(block_string[block_position_in_segment][0])))
		{
			temp_bit_max_dc = TOPOSITIVE(block_string[block_position_in_segment][0]);
		}
		
		if(block_string[block_position_in_segment][0] > dc_max)
		{
			dc_max = block_string[block_position_in_segment][0];
		}

		if(block_string[block_position_in_segment][0] > dc_min)
		{
			dc_min = block_string[block_position_in_segment][0];
		}

		signed long ac_bit_max_one_block = 0; 
		for(unsigned int i = 1; i < BLOCKSIZEIMAGE * BLOCKSIZEIMAGE; ++i)
		{
			unsigned int abs_ac = 0;
			abs_ac = TOPOSITIVE(block_string[block_position_in_segment][i]);
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
		unsigned long new_value = conversion_twos_complement (block_string[i][0], bit_depth_dc);
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
		for(unsigned int i = 0; i < blocks_in_segment; ++i)
		{
			shifted_dc[i]; // print to bit stream	
		}

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
	for (unsigned int i; i < (N -1); ++i)
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
		for (unsigned int i = 0; i < num_add_bit_planes; ++i)
		{
			for(unsigned int k = 0; k < blocks_in_segment; ++k)
			{
				// TODO print dc_remainer[k] >> (quantization_factor - i -1)
			}
		}

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
	// now output coded ( or uncoded) values

	for (unsigned int i = star_index; i < star_index + gaggles; ++i)
	{
		if ((min_k == uncoded_flag) || (i==0))
		{
			// print bits dc_mapper[i]; N times// uncoded
		}
		else
		{
			// prints 1 (dc_mapper[i] >> min_k) +1
		}
	}
	// if we have coded samples, then we also have to send the second part
	if (min_k != uncoded_flag)
	{	
		int max_value = star_index > 1 ? star_index : 1; // max operation (づ｡◕‿‿◕｡)づ
		for(unsigned int i = max_value; i < star_index + gaggles; ++i)
		{
			// print_out dc_mapper[i], min_k
		}
	}
	
}

void acb_pe_encoding(header_struct *header_pointer, BOOL *segment_full, unsigned int blocks_in_segment, unsigned short *bit_max_in_segment, unsigned char quantization_factor, unsigned short *dc_remainer)
{
	unsigned char bit_plane_data;
	// 4.4 
	if (header_pointer->header.part1.bit_depth_ac_5bits != 0) // if is not 0 we cantinue processing if bit depth is 0 is not encoded. 4.4 a)
	{
		if (header_pointer->header.part1.bit_depth_ac_5bits == 1) // this means that ac is or ether 1 or 0 so we can encoded in one bit 4.4 b)
		{
			for(unsigned int i = 0; i < blocks_in_segment; ++i)
			{
				// Print bits of bit_max_in_segment
			}

		}
		else // otherwose, the secuence of BitDepthAC whoud be codeed simililar to DC 4.4 c)
		{
			ac_depth_encoder(header_pointer, blocks_in_segment, bit_max_in_segment, segment_full);
		}

		// 3.2 tree scaning

		for (unsigned int bit_plane = header_pointer->header.part1.bit_depth_ac_5bits; bit_plane > 0; --bit_plane)
		{
			bit_plane_data = bit_plane;
			if ((header_pointer->header.part2.bit_plane_stop_5bits == bit_plane) && (header_pointer->header.part1.part_2_flag == TRUE)){
				return; // exit part
			}

			// statage 0
			// encode the DC component (single bit only)
			// if ( B && ( A || (C && D) ) ){}
			if ((bit_plane <= quantization_factor) && ((header_pointer->header.part4.dwt_type == INTEGER_WAVELET) || ((quantization_factor > header_pointer->header.part4.custom_wt_LL3_2bits) && (header_pointer->header.part4.custom_wt_LL3_2bits < bit_plane))))
			{
				// print bits dc that have beeb shifted out
				for (unsigned int i = 0; i < blocks_in_segment; ++i)
				{
					// print bits ((dc_remainer[i] >> (bit_plane -1)) & 0x01) 1 bit
				}

			}
			if(*segment_full == TRUE)
			{
				return;
			}
			//block_scan_encode();
		
		
		
		}

		

	}
}

void ac_depth_encoder(header_struct *header_pointer, unsigned int blocks_in_segment, unsigned short *bit_max_in_segment,  BOOL *segment_full)
{
	signed int gaggle_start_index;
	unsigned char gaggles;
	unsigned char max_k;
	unsigned char id_lengh;
	unsigned char N = 0;

	while ( header_pointer->header.part1.bit_depth_ac_5bits >> N > 0) // get the numer of bits (づ｡◕‿‿◕｡)づ
	{
		++N;
	}
	unsigned short *mapped_ac = NULL;
	mapped_ac = (unsigned short*)calloc(blocks_in_segment, sizeof(unsigned short));
	dpcm_ac_mapper(header_pointer, blocks_in_segment, N, bit_max_in_segment, mapped_ac);
	// 4.3.2.7
	if (N == 2)
	{
		max_k = 0;
		id_lengh = 0;
	}
	else if (N <= 4)
	{
		max_k = 2;
		id_lengh = 2;
	}
	else if (N <= 5)
	{
		max_k = 6;
		id_lengh = 3;
	}
	else
	{
		printf("BPE ERROR");
		max_k = 0;
		id_lengh = 0;
	}
	
	gaggle_start_index = 0;
	while ( gaggle_start_index < blocks_in_segment)
	{
		gaggles = GAGGLE_SIZE > (blocks_in_segment - gaggle_start_index) ? (blocks_in_segment - gaggle_start_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
		ac_gaggle_encoding(header_pointer,mapped_ac,gaggle_start_index,gaggles,max_k, id_lengh,N);
		if (*segment_full == TRUE)
		{
			return;
		}
		gaggle_start_index += gaggles;
	}
	
	



}
// similar to the DC mapper in the 
void dpcm_ac_mapper(header_struct *header_pointer, unsigned int blocks_in_segment, unsigned char N, unsigned short *bit_max_in_segment,unsigned short *mapped_ac)
{

	short *diff_ac = NULL;
	short theta = 0;
	int x_min = 0;
	int x_max = ((1 << N) - 1);

	diff_ac = (short*)calloc(blocks_in_segment, sizeof(short));
	diff_ac[0] = bit_max_in_segment[0];
	mapped_ac[0] = bit_max_in_segment[0];
	for (unsigned int i = 1; i < blocks_in_segment; ++i)
	{
		diff_ac[i] = bit_max_in_segment[i] - bit_max_in_segment[i - 1]; // get the diference between elements
	}
	for (unsigned int i = 1; i < blocks_in_segment; ++i)
	{
		theta = (bit_max_in_segment[i-1] - x_min) > (x_max - bit_max_in_segment[i -1]) ? (x_max - bit_max_in_segment[i -1]) : (bit_max_in_segment[i-1] - x_min); // min operation (づ｡◕‿‿◕｡)づ
		// 4.3.2.4 ecuations 18 and 19
		if (diff_ac[i] >= 0 && diff_ac[i] <= theta)
		{
			mapped_ac[i] = 2 * diff_ac[i];
		}
		else if (diff_ac[i] < 0 && diff_ac[i] >= -theta)
		{
			mapped_ac[i] = - 2 * diff_ac[i] - 1;
		}
		else
		{
			mapped_ac[i] = theta + abs(diff_ac[i]);
		}
		
	}
	free(diff_ac);
}

void ac_gaggle_encoding(header_struct *header_pointer, unsigned short *mapped_ac, signed int gaggle_start_index, unsigned char gaggles, unsigned char max_k, unsigned char id_lengh, unsigned char N)
{
	int min_k = 0;
	unsigned long min_bits = 0xFFF;
	unsigned long total_bits = 0;

	unsigned char uncoded_flag = ~0;

	// determine code choice min_k to use, using brute-force or optimiun calculation
	// 4.3.2.11
	if(header_pointer->header.part3.opt_ac_select == TRUE) // Brute force (╯°□°）╯︵ ┻━┻ 
	{ 
		min_k = uncoded_flag;  // uncoded option used, unless we find a better option below
		for (unsigned int k = 0; k <= max_k; ++k)
		{
			if (gaggle_start_index == 0)
			{
				total_bits = N; // cost of uncoded first sample
			}
			else
			{
				total_bits = 0;
			}
			signed int max_value = gaggle_start_index > 1 ? gaggle_start_index : 1; // max operation (づ｡◕‿‿◕｡)づ
			for (unsigned int i = max_value; i < gaggle_start_index + gaggles; ++i)
			{
				total_bits += ((mapped_ac[i] >> k) + 1) +k;  // coded sample cost
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
		int J = gaggles;

		if (gaggle_start_index == 0)
		{
			J = gaggles - 1;
		}

		for (unsigned int i = gaggle_start_index; i < gaggle_start_index + gaggles; ++i)
		{
			delta += mapped_ac[i];
		}

		if (64*delta >= 23*J*(1 << N))
		{
			min_k = uncoded_flag ;   // indicate that uncoded option is to be used
		}
		else if (207 * J > 128 * delta)
		{
			min_k = 0;
		}
		else if ((long)(J * (1 << (N + 5))) <= (long)(128 * delta + 49 * J))
		{
			min_k = N -2;
		}
		else
		{
			min_k = 0;
			while ((long)(J * (1 << (min_k + 7))) <= (long)(128 * delta + 49 * J))
			{
				++min_k; 
			}
			--min_k;
			
		}
			
	}
	// ended the min_k extraction 

	// Print bits( min_K, id_length)

	// now output codded (or uncoded) values
	for (unsigned int i = 0; i < gaggle_start_index + gaggles; ++i)
	{
		if((min_k == uncoded_flag) || (i==0))
		{
			// Print bits( mapped_ac[i], N)  // uncoded
		}
		else
		{
			// Print bits( 1, (mapped_ac[i] >> min_k) + 1)  // codded
		}
		
	}
	// if we have coded samples, then we also have to send the second part
	if (min_k != uncoded_flag)
	{
		signed int max_value = gaggle_start_index > 1 ? gaggle_start_index : 1; // max operation (づ｡◕‿‿◕｡)づ
		for (unsigned int i = max_value; i < gaggle_start_index + gaggles; ++i)
		{
			// Print bits(mapped_ac[i],  min_k) 
		}
	}
	


}

// 4.5.1
void block_scan_encode(long ** block_string,unsigned int block_position, header_struct *header_pointer,unsigned int bit_plane, unsigned int blocks_in_segment,unsigned short *bit_max_in_segment)
{
	unsigned char si = 0;
	unsigned long block_index = 0;
	long * block = NULL;
	for ( unsigned long block_seq = 0; block_seq < blocks_in_segment; ++block_seq)
	{
		si = 0;
		if (bit_max_in_segment[block_seq] < bit_plane){
			continue; // skip 
		}
		block_index = block_seq;
		block = block_string[block_seq + block_position]; // retrive a block
		// Start the Coding stages 1 - 3
		// parent codding there is 3 parents p0, p1, p2
		for (unsigned int i = 0; i < 3; ++i)
		{
			// check if is integer 
			if (header_pointer->header.part4.dwt_type == INTEGER_WAVELET)
			{
				// check if we have a custom bit plane
				if (((i == 0) && (header_pointer->header.part4.custom_wt_HL3_2bits >= bit_plane)) 
				|| ((i== 1) && (header_pointer->header.part4.custom_wt_LH3_2bits >= bit_plane)) 
				|| ((i==2) && header_pointer->header.part4.custom_wt_HH3_2bits))
				{
					continue; // ignore
				}
			}


		}


	}
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
        //printf("%02X ",data_array[i]);
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
						unsigned long value = segment_data.dc_mapper[i];
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
	unsigned int last_position_number_bits = total_bits % 8;
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

void clean(DataObject *device_object){
	free(device_object->data_array);
	delete device_object;
}