
/**
 * \file util_data_files.c 
 * \brief Benchmark #1.1 random data generation.
 * \author Ivan Rodriguez (BSC)
 */


#include "util_data_files.h"


int load_data_from_file(

	char * filename,
	unsigned int *data, 
	
	unsigned int j_blocksize,
    unsigned int r_samplesInterval
	)
{
    FILE *file = fopen(filename, "rb");
    if(file == NULL)
    {
        printf("error: failed to open file: %s\n", filename);
        return FILE_LOADING_ERROR;
    }
    // read each of the elements in the file
    for (int i = 0; i < j_blocksize * r_samplesInterval; i++)
    {
        fread(&data[i], sizeof(short int), 1, file);
    }
    fclose(file);
    return FILE_LOADING_SUCCESS;
}


int store_data_to_file(
    char * filename,
    unsigned char *data,
    unsigned int num_elements
    )
{

        FILE *fp;
        fp = fopen(filename, "wb");
        fwrite(data, 1, num_elements, fp); 
        fclose(fp);
        return FILE_LOADING_SUCCESS;
}