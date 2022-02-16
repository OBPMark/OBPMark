/**
 * \file util_file.c
 * \brief File reader for benchmark #3.1
 * \author Marc Solé Bonet (BSC)
 */
#include "util_file.h"
#include <stdio.h>

#define ERROR_FILELOAD	-2
#define ERROR_FILEREAD	-3
#define READ_SUCCESS     1

/* Functions */ 

int get_file_data(const char *file, const char *key, uint8_t *input_data, uint8_t *cypher_key, unsigned int data_length, unsigned int key_length)
{
    /* Open file, if error return ERROR_FILELOAD */
    FILE* fdata = fopen(file, "rb");
    if (!fdata) {
        printf("error: Could not open %s", file);
        return ERROR_FILELOAD;
    }

    /* Write length bytes from fdata into buffer */
    size_t data = fread(input_data, sizeof(uint8_t), data_length, fdata);
    if (data <= 0) {
        printf("error: Could not read from %s", file);
        return ERROR_FILEREAD;
    }
    fclose(fdata);

    FILE* fkey = fopen(key, "rb");
    if (!fkey) {
        printf("error: Could not open %s", key);
        return ERROR_FILELOAD;
    }
    data = fread(cypher_key, sizeof(uint8_t), key_length, fkey);
    if (data <= 0) {
        printf("error: Could not read from %s", file);
        return ERROR_FILEREAD;
    }
    fclose(fkey);

    return READ_SUCCESS;

}
