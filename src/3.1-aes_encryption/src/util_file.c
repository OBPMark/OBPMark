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

int get_file_data(const char *file, uint8_t *input_data, uint8_t *cypher_key, unsigned int data_length, unsigned int key_length)
{
    /* Open file, if error return ERROR_FILELOAD */
    FILE* fdata = fopen(file, "rb");
    if (!fdata) return ERROR_FILELOAD;

    /* Write length bytes from fdata into buffer */
    size_t key = fread(cypher_key, sizeof(uint8_t), key_length, fdata);
    if (!key) return ERROR_FILEREAD;
    size_t data = fread(input_data, sizeof(uint8_t), data_length, fdata);
    if (!data) return ERROR_FILEREAD;

    return READ_SUCCESS;

}
