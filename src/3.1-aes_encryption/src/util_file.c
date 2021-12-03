/**
 * \file util_file.c
 * \brief File reader for benchmark #3.1
 * \author Marc Solé Bonet (BSC)
 */
#include "util_file.h"
#include <stdio.h>

#define ERROR_FILELOAD	-2
#define ERROR_FILEREAD	-3

//FILE open_file(const char *file_name)
//{
//    return fopen(file_name, "rb");
//}

/* Functions */ 

int get_file_data(const char *file, unsigned int length, uint8_t *buffer)
{
    /* Open file, if error return ERROR_FILELOAD */
    FILE* fdata = fopen(file, "rb");
    if (!fdata) return ERROR_FILELOAD;

    /* Write length bytes from fdata into buffer */
    size_t read = fread(buffer, sizeof(uint8_t), length, fdata);
    if (!read) return ERROR_FILEREAD;

    return read;

}
