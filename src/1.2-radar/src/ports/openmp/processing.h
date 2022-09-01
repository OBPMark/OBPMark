/** 
 * \brief OBPMark "Radar image generation" processing task and image kernels.
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

uint32_t next_power_of2(uint32_t n);
void SAR_focus(radar_data_t *data);



#endif // PROCESSING_H_

