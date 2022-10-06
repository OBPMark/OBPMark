#ifndef OUTPUTFORMATUTILS_H
#define OUTPUTFORMATUTILS_H

#include <cstddef>
#include <cstdlib>
#include <cstdio>

#define BYTE_SIZE 8

struct SegmentBitStreamByte
{
    unsigned int bit_size;
    unsigned int byte_4_value;
    struct SegmentBitStreamByte *next;
};


struct SegmentBitStream
{
    unsigned int segment_id;
    unsigned int num_total_bytes;
    struct SegmentBitStreamByte *first_byte;
    struct SegmentBitStreamByte *last_byte;

};


void write_to_the_output_segment(struct SegmentBitStream *segment_list, int word, unsigned int length, unsigned int segment_id);
void clean_segment_bit_stream(struct SegmentBitStream *segment_list, unsigned int number_segments);
void round_up_last_byte(struct SegmentBitStream *segment_bit_stream, unsigned int segment_id);
//void add_byte_to_segment_bit_stream(struct SegmentBitStream *segment_bit_stream, unsigned int segment_id, unsigned char value);

void print_segment_list(struct SegmentBitStream *segment_list, unsigned int num_segments);
void write_segment_list(struct SegmentBitStream *segment_list, unsigned int num_segments, char *filename);
#endif // OUTPUTFORMATUTILS_H