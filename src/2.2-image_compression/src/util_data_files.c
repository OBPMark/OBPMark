/**
 * \file util_data_files.c 
 * \brief Benchmark #2.2 random data generation.
 * \author Ivan Rodriguez (BSC)
 */


#include "util_data_files.h"


int load_data_from_file(
	char * filename,
	compression_image_data_t * ccsds_data
	)
{
    int result = 0;
    FILE *file = fopen(filename, "rb");
    if(file == NULL)
    {
        printf("error: failed to open file: %s\n", filename);
        return FILE_LOADING_ERROR;
    }

    // select the right size for reading the file depending of the bit size
    if (ccsds_data->bit_size <= 8)
    {
        for (int i = 0; i < ccsds_data->h_size * ccsds_data->w_size; i++)
        {
            result = fread(&ccsds_data->input_image[i], sizeof(char), 1, file);
            if(result != 1)
            {
                printf("error: failed to read file: %s\n", filename);
                return FILE_LOADING_ERROR;
            }
        }
    }
    else if (ccsds_data->bit_size <= 16)
    {
        for (int i = 0; i < ccsds_data->h_size * ccsds_data->w_size; i++)
        {
            result = fread(&ccsds_data->input_image[i], sizeof(short int), 1, file);
            if(result != 1)
            {
                printf("error: failed to read file: %s\n", filename);
                return FILE_LOADING_ERROR;
            }
        }
    }
    else if (ccsds_data->bit_size <= 32)
    {
        for (int i = 0; i < ccsds_data->h_size * ccsds_data->w_size; i++)
        {
            result = fread(&ccsds_data->input_image[i], sizeof(unsigned int), 1, file);
            if(result != 1)
            {
                printf("error: failed to read file: %s\n", filename);
                return FILE_LOADING_ERROR;
            }
        }
    }
    else
    {
        printf("error: bit size not supported\n");
        return FILE_LOADING_ERROR;
    }
    // read each of the elements in the file

    // check if padding is needed
    if (ccsds_data->pad_rows != 0 || ccsds_data->pad_columns != 0)
    {
        // add padding
        for(unsigned int i = 0; i < ccsds_data->pad_rows ; i++)
        {
            for(unsigned int j = 0; j < ccsds_data->h_size + ccsds_data->pad_columns; j++)
                ccsds_data->input_image[(i + ccsds_data->h_size) * ccsds_data->w_size + j] = ccsds_data->input_image[(ccsds_data->h_size - 1) * ccsds_data->w_size + j];
        }

        for(unsigned int i = 0; i < ccsds_data->pad_columns ; i++)
        {
            for(unsigned int j = 0; j < ccsds_data->w_size + ccsds_data->pad_rows ; j++)
                ccsds_data->input_image[(j) * ccsds_data->w_size + (i + ccsds_data->w_size)] = ccsds_data->input_image[(j) * ccsds_data->w_size + (ccsds_data->w_size - 1)];
        }
    }
   
    fclose(file);
    return FILE_LOADING_SUCCESS;
}


int store_data_to_file(
    char * filename,
    compression_image_data_t * ccdsd_data
    )
{
 return FILE_STORAGE_SUCCESS;   
}