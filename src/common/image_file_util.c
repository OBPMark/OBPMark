/**
 * \file image_file_util.c
 * \brief General I/O functions for image files.
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info, see the LICENSE file in the root directory.
 */
#include "image_file_util.h"
#include <stdio.h> 


/* 2D buffers */

# ifdef OBPMARK_FRAME_DATA_2D
/* Write data to file */

int write_frame(char filename[], void **f, unsigned int width, unsigned int height, int data_width, uint8_t verbose)
{
	FILE *framefile;
	size_t bytes_written;
	size_t bytes_expected = height * data_width;
	size_t bytes_total=0;
	unsigned int x;

	framefile = fopen(filename, "wb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<width; x++)
	{
		bytes_written = data_width * fwrite(f[x], data_width, height, framefile);
		bytes_total += bytes_written;
		if(bytes_written != bytes_expected) {
			printf("error: writing file: %s, failed at col: %d, expected: %ld bytes, wrote: %ld bytes, total written: %ld bytes\n",
					filename, x, bytes_expected, bytes_written, bytes_total);
			return 0;
		}
	}

	fclose(framefile);
	if (verbose == 1)
	{
		printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*width));
	}
	
	return 1;
}

int write_frame8(char filename[], frame8_t *frame, uint8_t verbose)
{
	return write_frame(filename, (void**)(frame->f), frame->w, frame->h, (int)sizeof(uint8_t), verbose);
}

int write_frame16(char filename[], frame16_t *frame, uint8_t verbose)
{
	return write_frame(filename, (void**)(frame->f), frame->w, frame->h, (int)sizeof(uint16_t), verbose);
}

int write_frame32(char filename[], frame32_t *frame, uint8_t verbose)
{
	return write_frame(filename, (void**)(frame->f), frame->w, frame->h, (int)sizeof(uint32_t), verbose);
}

/* Read data from file */

int read_frame(char filename[], void **f, unsigned int width, unsigned int height, int data_width)
{
	FILE *framefile;
	unsigned int x;
	size_t bytes_read;
	size_t bytes_expected = height * data_width;
	size_t bytes_total=0;

	framefile = fopen(filename, "rb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<width; x++)
	{
		bytes_read = data_width * fread(f[x], data_width, height, framefile);
		bytes_total += bytes_read;
		if(bytes_read != bytes_expected) {
			printf("error: reading file: %s, failed at col: %d, expected: %ld bytes, read: %ld bytes, total read: %ld bytes\n",
					filename, x, bytes_expected, bytes_read, bytes_total);
			return 0;
		}
	}
	
	fclose(framefile);
	printf("Read %ld bytes from file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*width));
	return 1;
}

int read_frame8(char filename[], frame8_t *frame)
{
	return read_frame(filename, (void**)(frame->f), frame->w, frame->h, 1);
}

int read_frame16(char filename[], frame16_t *frame)
{
	return read_frame(filename, (void**)(frame->f), frame->w, frame->h, 2);
}

int read_frame32(char filename[], frame32_t *frame)
{
	return read_frame(filename, (void**)(frame->f), frame->w, frame->h, 4);
}

# else
/* 1D buffers */

/* Write data to file */

int write_frame8(char filename[], frame8_t *frame, uint8_t verbose)
{

	FILE *framefile;
	size_t bytes_written;
	static const int data_width = (int)sizeof(uint8_t);
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;
	unsigned int x;

	framefile = fopen(filename, "wb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_written = data_width * fwrite(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_written;
		if(bytes_written != bytes_expected) {
			printf("error: writing file: %s, failed at col: %d, expected: %ld bytes, wrote: %ld bytes, total written: %ld bytes\n",
					filename, x, bytes_expected, bytes_written, bytes_total);
			return 0;
		}
	}

	fclose(framefile);
	if (verbose == 1)
	{
		printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*frame->w));
	}
	return 1;
}

int write_frame16(char filename[], frame16_t *frame, uint8_t verbose)
{
	FILE *framefile;
	size_t bytes_written;
	static const int data_width = (int)sizeof(uint16_t);
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;
	unsigned int x;

	framefile = fopen(filename, "wb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_written = data_width * fwrite(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_written;
		if(bytes_written != bytes_expected) {
			printf("error: writing file: %s, failed at col: %d, expected: %ld bytes, wrote: %ld bytes, total written: %ld bytes\n",
					filename, x, bytes_expected, bytes_written, bytes_total);
			return 0;
		}
	}

	fclose(framefile);
	if (verbose == 1)
	{
		printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*frame->w));
	}
	return 1;
}

int write_frame32(char filename[], frame32_t *frame, uint8_t verbose)
{
	FILE *framefile;
	size_t bytes_written;
	static const int data_width = (int)sizeof(uint32_t);
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;
	unsigned int x;

	framefile = fopen(filename, "wb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_written = data_width * fwrite(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_written;
		if(bytes_written != bytes_expected) {
			printf("error: writing file: %s, failed at col: %d, expected: %ld bytes, wrote: %ld bytes, total written: %ld bytes\n",
					filename, x, bytes_expected, bytes_written, bytes_total);
			return 0;
		}
	}

	fclose(framefile);
	if (verbose == 1)
	{
		printf("Wrote %ld bytes to file: %s, (expected %ld bytes)\n", bytes_total, filename, (bytes_expected*frame->w));
	}
	return 1;
}

/* Read data from file */


int read_frame8(char filename[], frame8_t *frame)
{
	FILE *framefile;
	unsigned int x;
	static const uint8_t data_width = 1;
	size_t bytes_read;
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;

	framefile = fopen(filename, "rb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_read = data_width * fread(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_read;
		if(bytes_read != bytes_expected) {
			printf("error: reading file: %s, failed at col: %d, expected: %ld bytes, read: %ld bytes, total read: %ld bytes\n",
					filename, x, bytes_expected, bytes_read, bytes_total);
			return 0;
		}
	}
	return 1;
}

int read_frame16(char filename[], frame16_t *frame)
{
	FILE *framefile;
	unsigned int x;
	static const uint8_t data_width = 2;
	size_t bytes_read;
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;

	framefile = fopen(filename, "rb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_read = data_width * fread(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_read;
		if(bytes_read != bytes_expected) {
			printf("error: reading file: %s, failed at col: %d, expected: %ld bytes, read: %ld bytes, total read: %ld bytes\n",
					filename, x, bytes_expected, bytes_read, bytes_total);
			return 0;
		}
	}
	return 1;
}

int read_frame32(char filename[], frame32_t *frame)
{
	FILE *framefile;
	unsigned int x;
	static const uint8_t data_width = 4;
	size_t bytes_read;
	size_t bytes_expected = frame->h * data_width;
	size_t bytes_total=0;

	framefile = fopen(filename, "rb");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", filename);
		return 0;
	}

	for(x=0; x<frame->w; x++)
	{
		bytes_read = data_width * fread(frame->f + (frame->h * x), data_width, frame->h, framefile);
		bytes_total += bytes_read;
		if(bytes_read != bytes_expected) {
			printf("error: reading file: %s, failed at col: %d, expected: %ld bytes, read: %ld bytes, total read: %ld bytes\n",
					filename, x, bytes_expected, bytes_read, bytes_total);
			return 0;
		}
	}
	return 1;
}

# endif
