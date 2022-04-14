/**
 * \file procesing.h
 * \brief Benchmark #1.1 CUDA processing header.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#ifndef PROCESSING_CUDA_H_
#define PROCESSING_CUDA_H_


/** 
 * \brief This function will apply Bias offset correction and Bad pixel correction to a incoming frame.
 */
__global__
void f_offset(
	uint16_t *frame,
	const uint16_t *offsets,
    const int size	
	);

/** 
 * \brief This function will apply  Bad pixel correction to a incoming frame.
 */
__global__
void f_mask_replace(
	uint16_t *frame,
	const uint8_t *mask,
    const unsigned int width,
    const unsigned int height
	);

/** 
 * \brief This function will apply to the current frame Radiation scrubbing.Frame_i_x are the four frames that will be used to calculate the new frame from the buffer frame.
 */
__global__
void f_scrub(
	uint16_t *frame,
	uint16_t *frame_i_0,
    uint16_t *frame_i_1,
    uint16_t *frame_i_2,
    uint16_t *frame_i_3,
    const unsigned int width,
    const unsigned int height
	);

/**
 * \brief Multiply a frame by a gain frame, pixel by pixel.
 */
__global__
void f_gain(
	uint16_t *frame,
	uint16_t *gains,
    const int size	
	);

/** 
 * \brief 2x2 binning and co-add pixels in a frame into a sum frame. 
 */
__global__
void f_2x2_bin_coadd(
	uint16_t *frame,
	uint32_t *sum_frame,
    const unsigned int width,
    const unsigned int height,
    const unsigned int lateral_stride
	);


#endif // PROCESSING_CUDA_H_
