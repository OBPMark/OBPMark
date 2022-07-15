/**
 * \file util_data_files.c 
 * \brief Benchmark #1.2 random data generation.
 * \author Marc Sole Bonet (BSC)
 */


#include "util_data_files.h"

#define MAX_FILENAME_LENGTH 256
#define DEFAULT_INPUT_FOLDER "../../data/input_data/1.2-radar"

int read_params(char filename[], radar_params_t *array);

int load_params_from_file(
        radar_params_t *params, 
        unsigned int height,
        unsigned int width,
        char *input_folder
        )
{

    bool default_input_folder = false;

    // check if the input folder is empty
    if(strcmp(input_folder,"") == 0)
    {
        // if it is empty, use the default folder
        default_input_folder = true;
    }

    /* Open parameters */
    char params_path[256];
    if (default_input_folder)
    {
        sprintf(params_path,"%s/%d_%d/1.2-radar-params_%d_%d.bin",DEFAULT_INPUT_FOLDER, height, width, height, width);
    }
    else
    {
        sprintf(params_path,"%s/1.2-radar-params_%d_%d.bin",input_folder,height,width);
    }
    if(!read_params(params_path, params)) return FILE_LOADING_ERROR;
    return FILE_LOADING_SUCCESS;
}



int read_params(char filename[], radar_params_t *array)
{
    FILE *framefile;
    static const uint8_t data_width = sizeof(radar_params_t);
    size_t bytes_read;
    size_t bytes_expected = data_width;

    framefile = fopen(filename, "rb");
    if(framefile == NULL) {
        printf("error: failed to open file: %s\n", filename);
        return 0;
    }
    bytes_read = data_width * fread(array, data_width, 1, framefile);
    if(bytes_read != bytes_expected) {
        printf("error: reading file: %s, failed, expected: %ld bytes, read: %ld bytes\n",
                filename, bytes_expected, bytes_read);
        return 0;
    }
    return 1;
}

int load_data_from_files(
        framefp_t *input_data, 
        radar_params_t *params, 
        unsigned int height, 
        unsigned int width, 
        char *input_folder
        )
{

    bool default_input_folder = false;

    /* Load data from files */
    // check if the input folder is empty
    if(strcmp(input_folder,"") == 0)
    {
        // if it is empty, use the default folder
        default_input_folder = true;
    }

    /* open input data */
    char data_path[256];
    unsigned int lines = pow(2,ceil(log2((float)params->asize+1)));
    for (int i = 0; i < params->npatch; i++){
        if (default_input_folder)
            sprintf(data_path,"%s/%d_%d/1.2-radar-patch_%d_%d_%d.bin",DEFAULT_INPUT_FOLDER, height, width, i, lines, width);
        else
            sprintf(data_path,"%s/1.2-radar-patch_%d_%d_%d.bin",input_folder, i, lines, width);

        if(!read_framefp(data_path, &input_data[i])) return FILE_LOADING_ERROR;
    }

    return FILE_LOADING_SUCCESS;
}

