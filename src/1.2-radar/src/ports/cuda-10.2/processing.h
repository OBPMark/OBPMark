/** 
 * \brief OBPMark "Image corrections and calibrations." processing task and image kernels.
 * \file processing.h
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#ifndef PROCESSING_H_
#define PROCESSING_H_

#include "device.h"
#include "obpmark.h"
#include "obpmark_image.h" 
#include "obpmark_time.h"

#include "math.h"
#include <cuComplex.h>

__global__ void printffDc();
__global__ void printfM();
__global__ void SAR_range_ref(float *rrf, radar_params_t *params, uint32_t nit);
__global__ void SAR_DCE(float *data, radar_params_t *params, float const_k);
__global__ void SAR_rcmc_table(radar_params_t *params, uint32_t *offsets);
__global__ void SAR_azimuth_ref(float *arf, radar_params_t *params);
__global__ void SAR_patch_processing(float *rrf, float *range_data, radar_params_t *params);
__global__ void SAR_ref_product(float *data, float *ref, uint32_t w, uint32_t h);
__global__ void SAR_rcmc(float *data, uint32_t *offsets, uint32_t width, uint32_t height);
__global__ void SAR_transpose(float *in, float *out, uint32_t in_width, uint32_t out_width, uint32_t nrows, uint32_t ncols);
__global__ void SAR_multilook(float *radar_data, float *image, radar_params_t *params, uint32_t width, uint32_t height);
__global__ void quantize(float *data, uint8_t *image, uint32_t width, uint32_t height);
__global__ void bin_reverse(float *data, unsigned int size, unsigned int group);
__global__ void fft_kernel(float* data, int loop, float wpr ,float wpi, unsigned int theads, unsigned int size);

#endif // PROCESSING_H_

