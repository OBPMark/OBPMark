#htvar kernel_code
void kernel coeff_regroup(global const int *A, global int *B, const unsigned int h_size, const unsigned int w_size)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    if ( i < h_size /8 && j < w_size/8)
    {
        // first band HH1 starts in i = (h_size>>1), j = (w_size>>1);
        const unsigned int i_hh =  (i * 4) + (h_size >>1);
        const unsigned int j_hh = (j * 4) + (w_size >>1);
        const unsigned int x_hh = ((i_hh - (h_size>>1)) << 1);
        const unsigned int y_hh = ((j_hh - (w_size>>1)) << 1);
        // second band HL1 starts in i = 0, j = (w_size>>1);
        const unsigned int i_hl =  i * 4;
        const unsigned int j_hl = (j * 4) + (w_size >>1);
        const unsigned int x_hl = (i_hl  << 1);
        const unsigned int y_hl = ((j_hl - (w_size>>1)) << 1);
        // third band LH1 starts in i = (h_size>>1), j = 0;
        const unsigned int i_lh =  (i * 4) + (h_size >>1);
        const unsigned int j_lh = j * 4;
        const unsigned int x_lh = ((i_lh - (h_size>>1)) << 1);
        const unsigned int y_lh = (j_lh<< 1);
        for (unsigned int p = 0; p < 4; ++p)
        {
            for (unsigned int k = 0; k < 4; ++k)
            {   
                B[(x_hh + (p+4)) * w_size + (y_hh + (k + 4))] = A[(i_hh + p - 4) * w_size + (j_hh + k -4)];
                B[(x_hl + p) * w_size + (y_hl + (k + 4))] = A[(i_hl + p) * w_size + (j_hl + k - 4)];
                B[(x_lh + (p+4)) * w_size + (y_lh + k)] = A[(i_lh + p - 4) * w_size + (j_lh + k)];
            }
        }
        // first band processed start second band
        // process hh2 band
        const unsigned int i_hh2 = (i * 2) + (h_size>>2);
        const unsigned int j_hh2 = (j * 2) + (w_size>>2);
        const unsigned int x_hh2 = ((i_hh2 - (h_size>>2)) <<2);
        const unsigned int y_hh2 = ((j_hh2 - (w_size>>2)) <<2);
        B[(x_hh2 + 2) * w_size + (y_hh2 + 2)] = A[(i_hh2) * w_size + (j_hh2)];
        B[(x_hh2 + 2) * w_size + (y_hh2 + 3)] = A[(i_hh2) * w_size + (j_hh2 +1)];
        B[(x_hh2 + 3) * w_size + (y_hh2 + 2)] = A[(i_hh2 + 1) * w_size + (j_hh2)];	
        B[(x_hh2 + 3) * w_size + (y_hh2 + 3)] = A[(i_hh2 + 1) * w_size + (j_hh2 + 1)];			
        
        // process hl2 band

        const unsigned int i_hl2 =  i * 2;
        const unsigned int j_hl2 = (j * 2) + (w_size>>2);
        const unsigned int x_hl2 = (i_hl2 <<2);
        const unsigned int y_hl2 = ((j_hl2 - (w_size>>2)) <<2);
        B[(x_hl2) * w_size + (y_hl2 + 2)] = A[(i_hl2) * w_size + (j_hl2)];	
        B[(x_hl2) * w_size + (y_hl2 + 3)] = A[(i_hl2) * w_size + (j_hl2 + 1)];	
        B[(x_hl2 + 1) * w_size + (y_hl2 + 2)] = A[(i_hl2 + 1) * w_size + (j_hl2)];		
        B[(x_hl2 + 1) * w_size + (y_hl2 + 3)] = A[(i_hl2 + 1) * w_size + (j_hl2 + 1)];				

        // process lh2 band
        const unsigned int i_lh2 =  (i * 2) + (h_size>>2);
        const unsigned int j_lh2 =  j * 2;
        const unsigned int x_lh2 = ((i_lh2 - (h_size>>2)) <<2);
        const unsigned int y_lh2 = (j_lh2<<2);
        B[(x_lh2 + 2) * w_size + (y_lh2)] = A[(i_lh2) * w_size + (j_lh2)];
        B[(x_lh2 + 2) * w_size + (y_lh2 + 1)] = A[(i_lh2) * w_size + (j_lh2 + 1)];			
        B[(x_lh2 + 3) * w_size + (y_lh2)] = A[(i_lh2+1) * w_size + (j_lh2)];			
        B[(x_lh2 + 3) * w_size + (y_lh2 + 1)] = A[(i_lh2 + 1) * w_size + (j_lh2 + 1)];

        // second band processed start thirt band
        const unsigned int x = (h_size>>3);
        // process hh3 band
        const unsigned int i_hh3 =  i + (h_size>>3);
        const unsigned int j_hh3 =  j + (w_size>>3);
        B[(((i_hh3 - x) <<3) + 1) * w_size + (((j_hh3 - (w_size>>3)) <<3) + 1)] = A[(i_hh3) * w_size + (j_hh3)];
        
        // process hl3 band
        const unsigned int i_hl3 =  i;
        const unsigned int j_hl3 =  j + (w_size>>3);
        B[(i_hl3 << 3) * w_size + (((j_hl3 - (w_size>>3)) <<3) + 1)] = A[(i_hl3) * w_size + (j_hl3)];
        
        // process lh3 band
        const unsigned int i_lh3 =  i + (h_size>>3);
        const unsigned int j_lh3 =  j;
        B[(((i_lh3 - x) <<3) + 1) * w_size + (j_lh3<<3)] = A[(i_lh3) * w_size + (j_lh3)];

        // process DC compoments
        B[(i<<3) * w_size + (j<<3)] = A[(i) * w_size + (j)];

    }
}

void kernel block_string_creation(global const int *A,global long *B, const unsigned int h_size, const unsigned int w_size)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    if(i < h_size/BLOCKSIZEIMAGE && j < w_size/BLOCKSIZEIMAGE)
    {
        for (unsigned int x = 0; x < BLOCKSIZEIMAGE; ++x)
            {
            for (unsigned int y =0; y < BLOCKSIZEIMAGE; ++y)
            {
                B[(i + j) * w_size + (x * BLOCKSIZEIMAGE + y)] = (long)(A[(i*BLOCKSIZEIMAGE +x) * w_size + (j*BLOCKSIZEIMAGE+y)]);
            }
        }
    }
}

void kernel transform_image_to_float(global const int *A, global float *B, unsigned int size)
{
    unsigned int i = get_global_id(0);
    if ( i < size)
    {
        B[i] = (float)(A[i]);
    }
    
}

void kernel copy_image_to_int(global const int *A, global float *B, unsigned int size)
{
    unsigned int i = get_global_id(0);
    if ( i < size)
    {
        B[i] = A[i];
    }
    
}

void kernel transform_image_to_int(global const float *A, global int *B, unsigned int size)
{
    unsigned int i = get_global_id(0);
    if ( i < size)
    {
        B[i] = A[i] >= 0 ? (int)(A[i] + 0.5) : (int)(A[i] - 0.5);
    }
    
}


void kernel wavelet_transform_low_int(global  int *A, global int *B, const int n, const int step, const int offset_vector){
    unsigned int size = n;
    unsigned int i = get_global_id(0);

    if (i < size){
        A = A + offset_vector;
        B = B + offset_vector;
        //printf("Value low %d %d: %d\n",step,i,A[(i * step)]);
        int sum_value_low = 0;
        if(i == 0){
            sum_value_low = A[0] - floor(- (B[(size * step)]/2.0) + (1.0/2.0));
        }
        else
        {
            sum_value_low = A[(2 * i) * step] - floor( - (( B[(i * step) + (size * step) -(1 * step)] +  B[(i * step) + (size*step)])/ 4.0) + (1.0/2.0) );
        }
        
        B[(i * step)] = sum_value_low;
        //printf("sum_value_low: %d: %d\n",i,sum_value_low);
    }
}
void kernel wavelet_transform_int(global  int *A, global int *B, const int n, const int step, const int offset_vector){
    unsigned int size = n;
    unsigned int i = get_global_id(0);
   
    if (i < size){
        A = A + offset_vector;
        B = B + offset_vector;
        //printf("Value %d %d: %d\n",step,i,A[(i * step)]);
        int sum_value_high = 0;
        // specific cases
        if(i == 0){
            sum_value_high = A[1 * step] - floor( ((9.0/16.0) * (A[0] + A[2* step])) - ((1.0/16.0) * (A[2* step] + A[4* step])) + (1.0/2.0));
        }
        else if(i == size -2){
            sum_value_high = A[ (2*size  - 3) * step] - floor( ((9.0/16.0) * (A[(2*size -4) * step] + A[(2*size -2)*step])) - ((1.0/16.0) * (A[(2*size - 6)* step] + A[(2*size - 2) * step])) + (1.0/2.0));
        }
        else if(i == size - 1){
            sum_value_high = A[(2*size - 1)* step] - floor( ((9.0/8.0) * (A[(2*size  -2) * step])) -  ((1.0/8.0) * (A[(2*size  - 4)* step ])) + (1.0/2.0));
        }
        else{
            // generic case
            sum_value_high = A[(2*i  +1)* step] - floor( ((9.0/16.0) * (A[(2*i)* step ] + A[(2*i +2)* step])) - ((1.0/16.0) * (A[(2*i  - 2)* step] + A[(2*i  + 4)* step])) + (1.0/2.0));
        }
        
        //store
        B[(i * step)+(size * step)] = sum_value_high;
        //printf("sum_value_low: %d: %d\n",i,sum_value_high);

        //__syncthreads();
        // low_part
        //for (unsigned int i = 0; i < size; ++i){
        
        //}
    }

}

void kernel wavelet_transform_float(global  float *A,global float *B, const int n, global const float *lowpass_filter,global const float *highpass_filter, const int step, const int offset_vector){
    unsigned int size = n;
    unsigned int i = get_global_id(0);

    unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
    int gi_end = HIGHPASSFILTERSIZE / 2;
    if (i < size){
        A = A + offset_vector;
        B = B + offset_vector;
        float sum_value_low = 0;
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
			sum_value_low += lowpass_filter[hi + hi_end] * A[x_position * step];
			
        }
		// store the value
		B[i * step] = sum_value_low;
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
			sum_value_high += highpass_filter[gi + gi_end] * A[x_position * step];
		}
		// store the value
		B[(i * step) + (size * step)] = sum_value_high;

    }
}
							                             																		                                    
                                           																										                                    
#htendvar