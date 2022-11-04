/**
 * \file image_file_util.h
 * \brief General I/O functions for image files.
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#ifndef IMAGE_FILE_UTIL_H_
#define IMAGE_FILE_UTIL_H_

#include "obpmark_image.h"

int write_frame8(char filename[], frame8_t *frame, uint8_t verbose);
int write_frame16(char filename[], frame16_t *frame, uint8_t verbose);
int write_frame32(char filename[], frame32_t *frame, uint8_t verbose);
int write_framefp(char filename[], framefp_t *frame, uint8_t verbose);

int read_frame8(char filename[], frame8_t *frame);
int read_frame16(char filename[], frame16_t *frame);
int read_frame32(char filename[], frame32_t *frame);
int read_framefp(char filename[], framefp_t *frame);


#endif // IMAGE_FILE_UTIL_H_
