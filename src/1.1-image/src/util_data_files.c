/**
 * \file util_data_files.c 
 * \brief Benchmark #1.1 random data generation.
 * \author Ivan Rodriguez (BSC)
 */


#include "util_data_files.h"

#define MAX_FILENAME_LENGTH 256
#define DEFAULT_INPUT_FOLDER "../../data/input_data/1.1-image"


int load_data_from_files(

	frame16_t *input_frames,
	frame32_t *output_frames, 
	
	frame16_t *offset_map,
	frame8_t *bad_pixel_map,
	frame16_t *gain_map,

	unsigned int w_size,
	unsigned int h_size,
	unsigned int num_frames,
    char *input_folder
	)
{
    unsigned int frame_position; 
	unsigned int w_position;
	unsigned int h_position;
    unsigned int frame_array_position = 0;

    bool default_input_folder = false;


    /* Load data from files */
    // check if the input folder is empty
    if(strcmp(input_folder,"") == 0)
    {
        // if it is empty, use the default folder
        default_input_folder = true;
    }
    /* open offset map */
    // create the offset map path base on the w_size and h_size
    char offset_map_path[256];
    if (default_input_folder)
    {
        sprintf(offset_map_path,"%s/%d/1.1-image-offsets_%dx%d.bin",DEFAULT_INPUT_FOLDER,w_size,w_size,h_size);
    }
    else
    {
        sprintf(offset_map_path,"%s/1.1-image-offsets_%dx%d.bin",input_folder,w_size,h_size);
    }
    // init the offset map
    offset_map->w = w_size;
	offset_map->h = h_size;
    // read the binary file into the offset map
    if(!read_frame16(offset_map_path, offset_map)) return FILE_LOADING_ERROR;

 
    /* open bad pixel map */
    // create the bad pixel map path base on the w_size and h_size
    char bad_pixel_map_path[256];
    if (default_input_folder)
    {
        sprintf(bad_pixel_map_path,"%s/%d/1.1-image-bad_pixels_%dx%d.bin",DEFAULT_INPUT_FOLDER,w_size,w_size,h_size);
    }
    else
    {
        sprintf(bad_pixel_map_path,"%s/1.1-image-bad_pixels_%dx%d.bin",input_folder,w_size,h_size);
    }
    // init the bad pixel map
    bad_pixel_map->w = w_size;
    bad_pixel_map->h = h_size;
    // read the binary file into the bad pixel map
    if(!read_frame8(bad_pixel_map_path, bad_pixel_map)) return FILE_LOADING_ERROR;


    /* open gain map */
    // create the gain map path base on the w_size and h_size
    char gain_map_path[256];

    if (default_input_folder)
    {
        sprintf(gain_map_path,"%s/%d/1.1-image-gains_%dx%d.bin",DEFAULT_INPUT_FOLDER,w_size,w_size,h_size);
    }
    else
    {
        sprintf(gain_map_path,"%s/1.1-image-gains_%dx%d.bin",input_folder,w_size,h_size);
    }
    // init the gain map
    gain_map->w = w_size;
    gain_map->h = h_size;
    // read the binary file into the gain map
    if(!read_frame16(gain_map_path, gain_map)) return FILE_LOADING_ERROR;

    /* open input frames */
    // create the input frames path base on the w_size and h_size
    char input_frames_path[256];

    // load scrubbed frames
    for (frame_position = 2; frame_position > 0; frame_position--)
    {
        if(default_input_folder)
        {
            sprintf(input_frames_path,"%s/%d/1.1-image-scrub_t-%d_%dx%d.bin",DEFAULT_INPUT_FOLDER,w_size,frame_position,w_size,h_size);
        }
        else
        {
            sprintf(input_frames_path,"%s/1.1-image-scrub_t-%d_%dx%d.bin",input_folder,frame_position,w_size,h_size);
        }
        // init the input frames
        input_frames[frame_array_position].w = w_size;
        input_frames[frame_array_position].h = h_size;
        // read the binary file into the input frames
        if(!read_frame16(input_frames_path, &input_frames[frame_array_position])) return FILE_LOADING_ERROR;
        frame_array_position++;
    }
    // load frames
    for (frame_position = 0; frame_position < num_frames - 4; frame_position++)
    {
        if(default_input_folder)
        {
            sprintf(input_frames_path,"%s/%d/1.1-image-data_%dx%d_frame%d.bin",DEFAULT_INPUT_FOLDER,w_size,w_size,h_size,frame_position);
        }
        else
        {
            sprintf(input_frames_path,"%s/1.1-image-data_%dx%d_frame%d.bin",input_folder,w_size,h_size,frame_position);
        }
        // init the input frames
        input_frames[frame_array_position].w = w_size;
        input_frames[frame_array_position].h = h_size;
        // read the binary file into the input frames
        if(!read_frame16(input_frames_path, &input_frames[frame_array_position])) return FILE_LOADING_ERROR;
        frame_array_position++;
    }
    // load srub frames
    for (frame_position = 1; frame_position < 3; frame_position++)
    {
        if(default_input_folder)
        {
            sprintf(input_frames_path,"%s/%d/1.1-image-scrub_t+%d_%dx%d.bin",DEFAULT_INPUT_FOLDER,w_size,frame_position,w_size,h_size);
        }
        else
        {
            sprintf(input_frames_path,"%s/1.1-image-scrub_t+%d_%dx%d.bin",input_folder,frame_position,w_size,h_size);
        }
        // init the input frames
        input_frames[frame_array_position].w = w_size;
        input_frames[frame_array_position].h = h_size;
        // read the binary file into the input frames
        if(!read_frame16(input_frames_path, &input_frames[frame_array_position])) return FILE_LOADING_ERROR;
        frame_array_position++;

    }

    return FILE_LOADING_SUCCESS;
}