#include "output_format_utils.h"


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



    /*// loop over all segments
    for (unsigned int i = 0; i < num_segments; i++)
    {
        struct SegmentBitStreamByte * tmp = segment_list[i].first_byte;
        while (tmp != NULL)
        {

            unsigned int bit_size = tmp->bit_size + remaining_bits;
            if (bit_size > 8)
            {
                unsigned int temp_bit_size = bit_size;
                // loop until the bit size is less than 8
                while (temp_bit_size > 8)
                {
                    unsigned char temp_byte = 0;
                    for (int i = 8 - 1; i >=0; --i)
                    {
                        temp_byte |= ((word >>i) & 0x1) << (7 - status->num_bits);
                    }
                    temp_bit_size = temp_bit_size - 8;
                 
                }
                // if the bit size is not 0, add the remaining bits to the next byte
                if (temp_bit_size != 0)
                {
                    remaining_bits_value = 0;
                    remaining_bits = temp_bit_size;
                }

            }
            else
            {
                if (remaining_bits == 0)
                {
                    remaining_bits_value = tmp->byte_4_value;
                    remaining_bits = bit_size;
                }
                else
                {
                    remaining_bits_value << tmp->bit_size;
                    remaining_bits_value += tmp->byte_4_value;
                    remaining_bits = bit_size;
                }
            }
            tmp = tmp->next;
            /*
            // get the size of each SegmentBitStreamByte
            unsigned int bit_size = tmp->bit_size + remaining_bits;
            if (bit_size == 32)
            {   
                // if remaining bits are 0, write the 4 byte to the file
                if (remaining_bits == 0)
                {
                    fwrite(&tmp->byte_4_value, sizeof(unsigned int), 1, output_file);
                }
                else
                {
                    // if remaining bits are not 0, concatenate the actual value with the remaining bits
                    unsigned int temp_value = tmp->byte_4_value;
                    temp_value <<= remaining_bits;
                    temp_value += remaining_bits_value;
                    fwrite(&temp_value, sizeof(unsigned int), 1, output_file);
                }
            }
            else if (bit_size >= 16)
            {
                if (remaining_bits == 0)
                {
                    // write the first 16 bits to the file and save the remaining bits
                    fwrite(&tmp->byte_4_value, sizeof(unsigned short), 1, output_file);
                    remaining_bits_value = tmp->byte_4_value >> 16;
                    remaining_bits = bit_size - 16;
                }
                else
                {
                    // if remaining bits are not 0, concatenate the actual value with the remaining bits and write the 16 bits to the file and save the remaining bits
                    unsigned int temp_value = tmp->byte_4_value;
                    temp_value <<= remaining_bits;
                    temp_value += remaining_bits_value;
                    fwrite(&temp_value, sizeof(unsigned short), 1, output_file);
                    remaining_bits_value = temp_value >> 16;
                    remaining_bits = bit_size - 16;
                }
            }
            else if (bit_size >= 8)
            {
                if (remaining_bits == 0)
                {
                    // write the first 8 bits to the file and save the remaining bits
                    fwrite(&tmp->byte_4_value, sizeof(unsigned char), 1, output_file);
                    remaining_bits_value = tmp->byte_4_value >> 8;
                    remaining_bits = bit_size - 8;
                }
                else
                {
                    // if remaining bits are not 0, concatenate the actual value with the remaining bits and write the 8 bits to the file and save the remaining bits
                    unsigned int temp_value = tmp->byte_4_value;
                    temp_value <<= remaining_bits;
                    temp_value += remaining_bits_value;

                    remaining_bits_value <<=  bit_size % 8;
                    // only create a mask to get the bit_size % 8 bits from temp_value
                    remaining_bits_value += temp_value 

                    fwrite(&temp_value, sizeof(unsigned char), 1, output_file);
                    remaining_bits_value = temp_value >> 8;
                    remaining_bits = bit_size - 8;
                }
            }
            else
            {
                // if remaining bits are not 0, concatenate the actual value with the remaining bits and save the remaining bits
                if (remaining_bits == 0)
                {
                    remaining_bits_value = tmp->byte_4_value;
                    remaining_bits = bit_size;
                }
                else
                {
                    remaining_bits_value << tmp->bit_size;
                    remaining_bits_value += tmp->byte_4_value;
                    remaining_bits = bit_size;
                }
            }
            
            
            tmp = tmp->next;
        }
    }*/
}