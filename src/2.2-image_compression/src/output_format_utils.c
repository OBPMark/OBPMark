#include "output_format_utils.h"

void get_total_number_of_bits(struct SegmentBitStream *segment_list, unsigned int segment, unsigned int *total_number_of_bits);


void add_byte_to_segment_bit_stream(struct SegmentBitStream *segment_bit_stream, unsigned int segment_id, int value, unsigned int bit_length)
{
   if(segment_bit_stream[segment_id].first_byte == NULL)
   {
        // Allocate memory for the first SegmentBitStreamByte
        segment_bit_stream[segment_id].first_byte = (struct SegmentBitStreamByte *)malloc(sizeof(struct SegmentBitStreamByte));
        segment_bit_stream[segment_id].first_byte->next = NULL;
        segment_bit_stream[segment_id].first_byte->byte_4_value = value;
        segment_bit_stream[segment_id].first_byte->bit_size = bit_length;
        segment_bit_stream[segment_id].last_byte = segment_bit_stream[segment_id].first_byte;
        segment_bit_stream[segment_id].num_total_bytes = 4;

   }
   else 
   {
        if (segment_bit_stream[segment_id].last_byte->next == NULL)
        {
            // Allocate memory for the next SegmentBitStreamByte
            segment_bit_stream[segment_id].last_byte->next = (struct SegmentBitStreamByte *)malloc(sizeof(struct SegmentBitStreamByte));
            segment_bit_stream[segment_id].last_byte->next->next = NULL;
            segment_bit_stream[segment_id].last_byte->next->byte_4_value = value;
            segment_bit_stream[segment_id].last_byte->next->bit_size = bit_length;
            segment_bit_stream[segment_id].last_byte = segment_bit_stream[segment_id].last_byte->next;
            segment_bit_stream[segment_id].num_total_bytes = segment_bit_stream[segment_id].num_total_bytes + 4;
        }
        else
        {
            printf("Error: SegmentBitStreamByte is already allocated\n");
            exit(1);
        }
        
   }
}

void round_up_last_byte(struct SegmentBitStream *segment_bit_stream, unsigned int segment_id)
{
  // first get the total number of bits in the segment
    unsigned int total_number_of_bits = 0;
    get_total_number_of_bits(segment_bit_stream, segment_id, &total_number_of_bits);
    // if the total number of bits is not a multiple of 8, then add all of the bits to the last byte
    if(total_number_of_bits % 8 != 0)
    {
        write_to_the_output_segment(segment_bit_stream, 0, BYTE_SIZE - (total_number_of_bits % 8) , segment_id);
    }
}

void clean_segment_bit_stream(struct SegmentBitStream *segment_list, unsigned int number_segments)
{
    for(unsigned int i = 0; i < number_segments; i++)
    {
          struct SegmentBitStreamByte *current_byte = segment_list[i].first_byte;
          struct SegmentBitStreamByte *next_byte = NULL;
          while(current_byte != NULL)
          {
                next_byte = current_byte->next;
                free(current_byte);
                current_byte = next_byte;
          }
    }
}

void write_to_the_output_segment(struct SegmentBitStream *segment_list, int word, unsigned int length, unsigned int segment_id)
{
    printf("Writing value %d with length %d\n", word, length);
   if (length != 0 )
   {
        /*if (length > 32)
        {
            int remaining_length = length - 32;
            add_byte_to_segment_bit_stream(segment_list, segment_id, word, remaining_length);
            length = 32;
        }*/
        add_byte_to_segment_bit_stream(segment_list, segment_id, word, length);
   }
   
   
}


void print_segment_list(struct SegmentBitStream *segment_list, unsigned int number_segments)
{
    // loop over all segments
    for (unsigned int i = 0; i < number_segments; i++)
    {
        // print segment linked list without destroying it
        struct SegmentBitStreamByte * tmp = segment_list[i].first_byte;
        printf("Segment %d: ", i);
        while (tmp != NULL)
        {
            printf("%d \n", tmp->byte_4_value);
            tmp = tmp->next;
        }
        
    }
}

void get_total_number_of_bits(struct SegmentBitStream *segment_list, unsigned int segment, unsigned int *total_number_of_bits)
{
    *total_number_of_bits = 0;
    struct SegmentBitStreamByte *tmp = segment_list[segment].first_byte;
    while(tmp != NULL)
    {
        *total_number_of_bits += tmp->bit_size;
        tmp = tmp->next;
    }
}

void writeWord(unsigned char *output_stream_segment, unsigned int *number_bits, unsigned int *number_bytes, unsigned int word, int bit_size)
{
    for (int i = bit_size - 1; i >=0; --i)
    {
        output_stream_segment[*number_bytes] |= ((word >>i) & 0x1) << (7 - (*number_bits));
        (*number_bits)++;
        // Check if the byte is complete
        if (*number_bits >= 8)
        {
            (*number_bits) = 0;
            (*number_bytes)++;
        }
    }
}

void write_segment_list(struct SegmentBitStream *segment_list, unsigned int num_segments, char *filename)
{

    unsigned int remaining_bits = 0;
    unsigned int remaining_bits_value = 0;
    // create the output file
    FILE *output_file = fopen(filename, "wb");

    for (unsigned int i = 0; i < num_segments; i++)
    {
        
        unsigned int segment_bit_length = 0;
        get_total_number_of_bits(segment_list, i, &segment_bit_length);
        // get the number of bytes needed to store the segment
        unsigned int num_bytes = segment_bit_length / 8;
        if (segment_bit_length % 8 != 0)
        {
            num_bytes++;
        }
        // create an array of bytes to store the segment
        unsigned char *segment_bytes = (unsigned char *)malloc(num_bytes  * sizeof(unsigned char));
        unsigned int *remaining_bits = (unsigned int *)malloc(sizeof(unsigned int));
        unsigned int *remaining_bytes = (unsigned int *)malloc(sizeof(unsigned int));
        *remaining_bits = 0;
        *remaining_bytes = 0;
        // initialize the array
        for (unsigned int j = 0; j < num_bytes; j++)
        {
            segment_bytes[j] = 0;
        }
        // go over the 
        struct SegmentBitStreamByte * tmp = segment_list[i].first_byte;
        while (tmp != NULL)
        {
            writeWord(segment_bytes, remaining_bits, remaining_bytes, tmp->byte_4_value, tmp->bit_size);
            tmp = tmp->next;
            
        }
        // write the segment to the output file
        for (unsigned int j = 0; j < num_bytes; j++)
        {
            fwrite(&segment_bytes[j], sizeof(unsigned char), 1, output_file);
        }
        free(segment_bytes);
        free(remaining_bits);
        free(remaining_bytes);
    }


}