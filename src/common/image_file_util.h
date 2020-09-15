/**
 * \file image_file_util.h
 * \brief General I/O functions for image files.
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#ifndef IMAGE_FILE_UTIL_H_
#define IMAGE_FILE_UTIL_H_

#include "image_util.h"

int write_frame(char filename[], void **f, unsigned int width, unsigned int height, int data_width);
int write_frame8(char filename[], frame8_t *frame);
int write_frame16(char filename[], frame16_t *frame);
int write_frame32(char filename[], frame32_t *frame);

int read_frame8(char filename[], frame8_t *frame);
int read_frame16(char filename[], frame16_t *frame);
int read_frame32(char filename[], frame32_t *frame);

#endif // IMAGE_FILE_UTIL_H_
