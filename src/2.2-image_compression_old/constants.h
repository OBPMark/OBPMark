#ifndef CONSTANTS_H
#define CONSTANTS_H

#define LEVELS_DWT 3

#define BLOCKSIZEBPE 64 // block size of the BPE

#define BLOCK_SIZE  16

#define NUMBER_STREAMS 4

#define BLOCKSIZEIMAGE 8

#define IMAGE_WIDTH_MIN 17
#define IMAGE_HEIGHT_MIN 17

#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9

typedef unsigned char BOOL;
#define TRUE 1
#define FALSE 0

#define INTEGER_WAVELET FALSE
#define FLOAT_WAVELET TRUE

#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9
static const float lowpass_filter_cpu[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const float highpass_filter_cpu[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};


// header constants
#define NUMERBLOCKPERSEGMENT 256
#define SEGMENTSIZE 1024
#define BITSPERFIXEL 1
#define TOPOSITIVE(a) ((a) >=0 ? (a): -(a))
#define GAGGLE_SIZE 16

// header structure

typedef struct HEADER_STRUCTURE_PART1
{
    BOOL start_img_flag : 1;
    BOOL end_img_flag : 1;
    unsigned char segment_count_8bits : 8;
    unsigned char bit_depth_dc_5bits : 5;
    unsigned char bit_depth_ac_5bits : 5;
    BOOL reserved : 1; // not used
    BOOL part_2_flag : 1;
    BOOL part_3_flag : 1;
    BOOL part_4_flag : 1;
    unsigned char pad_rows_3bits : 3;
    unsigned char reserved_5bits : 5; //00000

}HeaderPart1;

typedef struct HEADER_STRUCTURE_PART2
{
   unsigned long seg_byte_limit_27bits : 27;
   BOOL dc_stop : 1;
   unsigned char bit_plane_stop_5bits: 5;
   unsigned char stage_stop_2bits : 2;
   BOOL use_fill :1;
   unsigned char reserved_4bits : 4; //0000

}HeaderPart2;

typedef struct HEADER_STRUCTURE_PART3
{
    unsigned long seg_size_blocks_20bits : 20; //max number of blocks in a segment is limited to 2^20
    BOOL opt_dc_select : 1;
    BOOL opt_ac_select : 1;
    unsigned char reserved_2bits : 2; //00
}HeaderPart3;

typedef struct HEADER_STRUCTURE_PART4
{
   BOOL dwt_type :1; //type of DWT TRUE is float and FALSE is integer
   unsigned char reserved_2bits : 2; // 00
   BOOL signed_pixels : 1;
   unsigned char pixel_bit_depth_4bits : 4;
   unsigned long image_with_20bits : 20;
   BOOL transpose_img : 1;
   unsigned char code_word_length : 3; // in the original code it uses 2 bits and the next one is a reserved one. In the new standart this reserved is used
   BOOL custom_wt_flag :1;
   unsigned char custom_wt_HH1_2bits : 2;
   unsigned char custom_wt_HL1_2bits : 2;
   unsigned char custom_wt_LH1_2bits : 2;
   unsigned char custom_wt_HH2_2bits : 2;
   unsigned char custom_wt_HL2_2bits : 2;
   unsigned char custom_wt_LH2_2bits : 2;
   unsigned char custom_wt_HH3_2bits : 2;
   unsigned char custom_wt_HL3_2bits : 2;
   unsigned char custom_wt_LH3_2bits : 2;
   unsigned char custom_wt_LL3_2bits : 2;
   unsigned short reserved_11bits : 11;

}HeaderPart4;

typedef struct HEADER
{
    HeaderPart1 part1;
    HeaderPart2 part2;
    HeaderPart3 part3;
    HeaderPart4 part4;
}header_struct_base;

typedef union HEADERUNION {
    header_struct_base header;
    long data[5];
}header_struct;


// bit_output_data

typedef struct BLOCK_BIT_OUTPUT
{   
    // all pointers except specify have the length of total_blocks_per_segment
    short N; // value of N
    header_struct_base *header; // header of the block
    unsigned long *shifted_dc_1bit; // shifted DC values
    // DC ENCODE size total_blocks_per_segment
    int *min_k; // number of gaggles
    unsigned int bits_min_k;
    unsigned long *dc_mapper;
    int numaddbitplanes;
    unsigned short *dc_remainer;
    unsigned char quantization_factor;

}block_struct;


#endif