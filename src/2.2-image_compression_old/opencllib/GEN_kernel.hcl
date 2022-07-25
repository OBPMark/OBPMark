
std::string kernel_code = 
"void kernel coeff_regroup(global const int *A, global int *B, const unsigned int h_size, const unsigned int w_size)\n"
"{\n"
"const unsigned int i = get_global_id(0);\n"
"const unsigned int j = get_global_id(1);\n"
"if ( i < h_size /8 && j < w_size/8)\n"
"{\n"
"// first band HH1 starts in i = (h_size>>1), j = (w_size>>1);\n"
"const unsigned int i_hh =  (i * 4) + (h_size >>1);\n"
"const unsigned int j_hh = (j * 4) + (w_size >>1);\n"
"const unsigned int x_hh = ((i_hh - (h_size>>1)) << 1);\n"
"const unsigned int y_hh = ((j_hh - (w_size>>1)) << 1);\n"
"// second band HL1 starts in i = 0, j = (w_size>>1);\n"
"const unsigned int i_hl =  i * 4;\n"
"const unsigned int j_hl = (j * 4) + (w_size >>1);\n"
"const unsigned int x_hl = (i_hl  << 1);\n"
"const unsigned int y_hl = ((j_hl - (w_size>>1)) << 1);\n"
"// third band LH1 starts in i = (h_size>>1), j = 0;\n"
"const unsigned int i_lh =  (i * 4) + (h_size >>1);\n"
"const unsigned int j_lh = j * 4;\n"
"const unsigned int x_lh = ((i_lh - (h_size>>1)) << 1);\n"
"const unsigned int y_lh = (j_lh<< 1);\n"
"for (unsigned int p = 0; p < 4; ++p)\n"
"{\n"
"for (unsigned int k = 0; k < 4; ++k)\n"
"{\n"
"B[(x_hh + (p+4)) * w_size + (y_hh + (k + 4))] = A[(i_hh + p - 4) * w_size + (j_hh + k -4)];\n"
"B[(x_hl + p) * w_size + (y_hl + (k + 4))] = A[(i_hl + p) * w_size + (j_hl + k - 4)];\n"
"B[(x_lh + (p+4)) * w_size + (y_lh + k)] = A[(i_lh + p - 4) * w_size + (j_lh + k)];\n"
"}\n"
"}\n"
"// first band processed start second band\n"
"// process hh2 band\n"
"const unsigned int i_hh2 = (i * 2) + (h_size>>2);\n"
"const unsigned int j_hh2 = (j * 2) + (w_size>>2);\n"
"const unsigned int x_hh2 = ((i_hh2 - (h_size>>2)) <<2);\n"
"const unsigned int y_hh2 = ((j_hh2 - (w_size>>2)) <<2);\n"
"B[(x_hh2 + 2) * w_size + (y_hh2 + 2)] = A[(i_hh2) * w_size + (j_hh2)];\n"
"B[(x_hh2 + 2) * w_size + (y_hh2 + 3)] = A[(i_hh2) * w_size + (j_hh2 +1)];\n"
"B[(x_hh2 + 3) * w_size + (y_hh2 + 2)] = A[(i_hh2 + 1) * w_size + (j_hh2)];\n"
"B[(x_hh2 + 3) * w_size + (y_hh2 + 3)] = A[(i_hh2 + 1) * w_size + (j_hh2 + 1)];\n"
"// process hl2 band\n"
"const unsigned int i_hl2 =  i * 2;\n"
"const unsigned int j_hl2 = (j * 2) + (w_size>>2);\n"
"const unsigned int x_hl2 = (i_hl2 <<2);\n"
"const unsigned int y_hl2 = ((j_hl2 - (w_size>>2)) <<2);\n"
"B[(x_hl2) * w_size + (y_hl2 + 2)] = A[(i_hl2) * w_size + (j_hl2)];\n"
"B[(x_hl2) * w_size + (y_hl2 + 3)] = A[(i_hl2) * w_size + (j_hl2 + 1)];\n"
"B[(x_hl2 + 1) * w_size + (y_hl2 + 2)] = A[(i_hl2 + 1) * w_size + (j_hl2)];\n"
"B[(x_hl2 + 1) * w_size + (y_hl2 + 3)] = A[(i_hl2 + 1) * w_size + (j_hl2 + 1)];\n"
"// process lh2 band\n"
"const unsigned int i_lh2 =  (i * 2) + (h_size>>2);\n"
"const unsigned int j_lh2 =  j * 2;\n"
"const unsigned int x_lh2 = ((i_lh2 - (h_size>>2)) <<2);\n"
"const unsigned int y_lh2 = (j_lh2<<2);\n"
"B[(x_lh2 + 2) * w_size + (y_lh2)] = A[(i_lh2) * w_size + (j_lh2)];\n"
"B[(x_lh2 + 2) * w_size + (y_lh2 + 1)] = A[(i_lh2) * w_size + (j_lh2 + 1)];\n"
"B[(x_lh2 + 3) * w_size + (y_lh2)] = A[(i_lh2+1) * w_size + (j_lh2)];\n"
"B[(x_lh2 + 3) * w_size + (y_lh2 + 1)] = A[(i_lh2 + 1) * w_size + (j_lh2 + 1)];\n"
"// second band processed start thirt band\n"
"const unsigned int x = (h_size>>3);\n"
"// process hh3 band\n"
"const unsigned int i_hh3 =  i + (h_size>>3);\n"
"const unsigned int j_hh3 =  j + (w_size>>3);\n"
"B[(((i_hh3 - x) <<3) + 1) * w_size + (((j_hh3 - (w_size>>3)) <<3) + 1)] = A[(i_hh3) * w_size + (j_hh3)];\n"
"// process hl3 band\n"
"const unsigned int i_hl3 =  i;\n"
"const unsigned int j_hl3 =  j + (w_size>>3);\n"
"B[(i_hl3 << 3) * w_size + (((j_hl3 - (w_size>>3)) <<3) + 1)] = A[(i_hl3) * w_size + (j_hl3)];\n"
"// process lh3 band\n"
"const unsigned int i_lh3 =  i + (h_size>>3);\n"
"const unsigned int j_lh3 =  j;\n"
"B[(((i_lh3 - x) <<3) + 1) * w_size + (j_lh3<<3)] = A[(i_lh3) * w_size + (j_lh3)];\n"
"// process DC compoments\n"
"B[(i<<3) * w_size + (j<<3)] = A[(i) * w_size + (j)];\n"
"}\n"
"}\n"
"void kernel block_string_creation(global const int *A,global long *B, const unsigned int h_size, const unsigned int w_size)\n"
"{\n"
"const unsigned int i = get_global_id(0);\n"
"const unsigned int j = get_global_id(1);\n"
"if(i < h_size/BLOCKSIZEIMAGE && j < w_size/BLOCKSIZEIMAGE)\n"
"{\n"
"for (unsigned int x = 0; x < BLOCKSIZEIMAGE; ++x)\n"
"{\n"
"for (unsigned int y =0; y < BLOCKSIZEIMAGE; ++y)\n"
"{\n"
"B[(i + j) * w_size + (x * BLOCKSIZEIMAGE + y)] = (long)(A[(i*BLOCKSIZEIMAGE +x) * w_size + (j*BLOCKSIZEIMAGE+y)]);\n"
"}\n"
"}\n"
"}\n"
"}\n"
"void kernel transform_image_to_float(global const int *A, global float *B, unsigned int size)\n"
"{\n"
"unsigned int i = get_global_id(0);\n"
"if ( i < size)\n"
"{\n"
"B[i] = (float)(A[i]);\n"
"}\n"
"}\n"
"void kernel copy_image_to_int(global const int *A, global float *B, unsigned int size)\n"
"{\n"
"unsigned int i = get_global_id(0);\n"
"if ( i < size)\n"
"{\n"
"B[i] = A[i];\n"
"}\n"
"}\n"
"void kernel transform_image_to_int(global const float *A, global int *B, unsigned int size)\n"
"{\n"
"unsigned int i = get_global_id(0);\n"
"if ( i < size)\n"
"{\n"
"B[i] = A[i] >= 0 ? (int)(A[i] + 0.5) : (int)(A[i] - 0.5);\n"
"}\n"
"}\n"
"void kernel wavelet_transform_low_int(global const int *A, global int *B, const int n, const int step, const int offset_vector){\n"
"unsigned int size = n;\n"
"unsigned int i = get_global_id(0);\n"
"if (i < size){\n"
"A = A + offset_vector;\n"
"B = B + offset_vector;\n"
"int sum_value_low = 0;\n"
"if(i == 0){\n"
"sum_value_low = A[0] - (int)(- (B[(size * step)]/2.0) + (1.0/2.0));\n"
"}\n"
"else\n"
"{\n"
"sum_value_low = A[(2 * i) * step] - (int)( - (( B[(i * step) + (size * step) -(1 * step)] +  B[(i * step) + (size*step)])/ 4.0) + (1.0/2.0) );\n"
"}\n"
"B[(i * step)] = sum_value_low;\n"
"}\n"
"}\n"
"void kernel wavelet_transform_int(global const int *A, global int *B, const int n, const int step, const int offset_vector){\n"
"unsigned int size = n;\n"
"unsigned int i = get_global_id(0);\n"
"if (i < size){\n"
"A = A + offset_vector;\n"
"B = B + offset_vector;\n"
"int sum_value_high = 0;\n"
"// specific cases\n"
"if(i == 0){\n"
"sum_value_high = A[1 * step] - (int)( ((9.0/16.0) * (A[0] + A[2* step])) - ((1.0/16.0) * (A[2* step] + A[4* step])) + (1.0/2.0));\n"
"}\n"
"else if(i == size -2){\n"
"sum_value_high = A[ (2*size  - 3) * step] - (int)( ((9.0/16.0) * (A[(2*size -4) * step] + A[(2*size -2)*step])) - ((1.0/16.0) * (A[(2*size - 6)* step] + A[(2*size - 2) * step])) + (1.0/2.0));\n"
"}\n"
"else if(i == size - 1){\n"
"sum_value_high = A[(2*size - 1)* step] - (int)( ((9.0/8.0) * (A[(2*size  -2) * step])) -  ((1.0/8.0) * (A[(2*size  - 4)* step ])) + (1.0/2.0));\n"
"}\n"
"else{\n"
"// generic case\n"
"sum_value_high = A[(2*i  +1)* step] - (int)( ((9.0/16.0) * (A[(2*i)* step ] + A[(2*i +2)* step])) - ((1.0/16.0) * (A[(2*i  - 2)* step] + A[(2*i  + 4)* step])) + (1.0/2.0));\n"
"}\n"
"//store\n"
"B[(i * step)+(size * step)] = sum_value_high;\n"
"//__syncthreads();\n"
"// low_part\n"
"//for (unsigned int i = 0; i < size; ++i){\n"
"//}\n"
"}\n"
"}\n"
"void kernel wavelet_transform_float(global const float *A,global float *B, const int n, global const float *lowpass_filter,global const float *highpass_filter, const int step, const int offset_vector){\n"
"unsigned int size = n;\n"
"unsigned int i = get_global_id(0);\n"
"unsigned int full_size = size * 2;\n"
"int hi_start = -(LOWPASSFILTERSIZE / 2);\n"
"int hi_end = LOWPASSFILTERSIZE / 2;\n"
"int gi_start = -(HIGHPASSFILTERSIZE / 2 );\n"
"int gi_end = HIGHPASSFILTERSIZE / 2;\n"
"if (i < size){\n"
"A = A + offset_vector;\n"
"B = B + offset_vector;\n"
"float sum_value_low = 0;\n"
"for (int hi = hi_start; hi < hi_end + 1; ++hi){\n"
"int x_position = (2 * i) + hi;\n"
"if (x_position < 0) {\n"
"// turn negative to positive\n"
"x_position = x_position * -1;\n"
"}\n"
"else if (x_position > full_size - 1)\n"
"{\n"
"x_position = full_size - 1 - (x_position - (full_size -1 ));\n"
"}\n"
"// now I need to restore the hi value to work with the array\n"
"sum_value_low += lowpass_filter[hi + hi_end] * A[x_position * step];\n"
"}\n"
"// store the value\n"
"B[i * step] = sum_value_low;\n"
"float sum_value_high = 0;\n"
"// second process the Highpass filter\n"
"for (int gi = gi_start; gi < gi_end + 1; ++gi){\n"
"int x_position = (2 * i) + gi + 1;\n"
"if (x_position < 0) {\n"
"// turn negative to positive\n"
"x_position = x_position * -1;\n"
"}\n"
"else if (x_position >  full_size - 1)\n"
"{\n"
"x_position = full_size - 1 - (x_position - (full_size -1 ));\n"
"}\n"
"sum_value_high += highpass_filter[gi + gi_end] * A[x_position * step];\n"
"}\n"
"// store the value\n"
"B[(i * step) + (size * step)] = sum_value_high;\n"
"}\n"
"}\n"
;
