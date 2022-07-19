/** 
 * \brief OBPMark "Data compression algorithm." processing task and image kernels.
 * \file processing.h
 * \author Ivan Rodriguez-Ferrandez (BSC)
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "obpmark.h"
#include "device.h"
#include "obpmark_image.h" 
#include "obpmark_time.h"



#define min(x, y) (((x) < (y)) ? (x) : (y))


/* 
This struct stores the last position a zeroblock presented a 0 including the number of 0's it presented 
position = -1; <- invalid position
*/
struct ZeroBlockCounter 
{
    unsigned int counter;
    int position;
};
/* Processed struct that determines the number of 0 to set for a given block, -1 means that the block is not a zero mem block */
struct ZeroBlockProcessed
{
    int NumberOfZeros;
};

struct FCompressedData 
{
    unsigned int size;
    unsigned int* data;
    unsigned char CompressionIdentifier; 
    unsigned int CompressionIdentifierInternal;
};

/**
 * \brief CCSDS 121 preprocess 
 */
void preprocess_data(
	compression_data_t *compression_data,
	unsigned int *ZeroCounterPos,
	struct ZeroBlockCounter * ZeroCounter,
	unsigned int step
	);

/**
 * \brief CCSDS 121 zeroblock computation 
 */
void process_zeroblock(
	compression_data_t *compression_data,
	unsigned int *ZeroCounterPos,
	struct ZeroBlockCounter *ZeroCounter,
	struct ZeroBlockProcessed *ZBProcessed
	);


/**
 * \brief CCSDS 121 process block computation 
 */

void process_blocks(
	compression_data_t *compression_data,
	struct ZeroBlockProcessed *ZBProcessed,
	unsigned int step
	);









#endif // PROCESSING_H_