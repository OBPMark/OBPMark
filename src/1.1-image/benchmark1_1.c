/** 
 * \brief OBPMark "Image corrections and calibrations." 
 * \file benchmark1_1.c
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include <stdio.h>
#include <stdint.h>
#include "image_kernels.h"

#include "../common/timing.h"
#include "../common/image_util.h" 
#include "../common/image_mem_util.h" 
#include "../common/image_file_util.h" 

#define DATA_ROOT_DIR "../../data/images/"
#define DATA_CALIBRATION

typedef struct {
	frame16_t *frames;
	unsigned int num_frames; 
	unsigned int num_neigh; 

	frame16_t offsets;
	frame16_t gains; 
	frame8_t bad_pixels;

	frame32_t binned_frame; 
	frame32_t image;

	time_t t_test;
	time_t *t_frame;

#ifdef OBPMARK_VERBOSE_TIMING
	time_t *t_offset;
	time_t *t_badpixel;
	time_t *t_scrub;
	time_t *t_gain;
	time_t *t_binning;
	time_t *t_coadd;
#endif // OBPMARK_VERBOSE_TIMING
} benchmark_params_t;

int benchmark_1_1_alloc_buffers(benchmark_params_t *params, unsigned int frame_width, unsigned int frame_height)
{
	/* 
	 * PORTING:
	 * If dynamic memory allocation should not be used, please define global static buffers in this file and make references in *param here. 
	 */
	if(!frame16_alloc(p->offsets,	frame_width, frame_height)) return 0;
	if(!frame16_alloc(p->gains,	frame_width, frame_height)) return 0;
	if(!frame8_alloc(p->bad_pixels,	frame_width, frame_height)) return 0;
	
	return 1; 
}

int benchmark_1_1_load_calibration_data(benchmark_params_t *params)
{
	int read_frame8(char filename[], frame8_t *frame);
	int read_frame16(char filename[], frame16_t *frame);

	// FIXME add different file names per frame width / height

	if(!read_frame16("../../data/images/frame_offsets.bin", p->offsets)) {
		printf("error: could not open offsets file.\n");
		return 0;
	}
	if(!read_frame16("../../data/images/frame_gains.bin", p->offsets)) {
		printf("error: could not open gains file.\n");
		return 0;
	}
	if(!read_frame8("../../data/images/frame_bad_pixels.bin", p->offsets)) {
		printf("error: could not open bad pixels file.\n");
		return 0;
	}

	return 1;
}

void benchmark_1_1_proc_frame(benchmark_params_t *p, frame16_t *frame, unsigned int frame_i)
{
	/* [I]: Bias offset correction */
	T_START_VERBOSE(p->t_offset);
	f_offset(frame, p->offset_frame);
	T_STOP_VERBOSE(p->t_offset);

	/* [II]: Bad pixel correction */
	T_START_VERBOSE(p->t_badpixel);
	f_mask_replace(frame, p->bad_pixels);
	T_STOP_VERBOSE(p->t_badpixel);

	/* [III]: Radiation scrubbing */
	T_START_VERBOSE(p->t_scrub);
	f_scrub(frame, p->num_frames, p->num_neigh);
	T_STOP_VERBOSE(p->t_scrub);

	/* [IV]: Gain correction */
	T_START_VERBOSE(p->t_gain);
	f_gain(frame, p->gains);
	T_STOP_VERBOSE(p->t_gain);

	/* [V]: Spatial binning */
	T_START_VERBOSE(p->t_binning);
	f_2x2_bin(frame, p->binned_frame);
	T_STOP_VERBOSE(p->t_binning);
	
	/* [VI]: Co-adding frames */
	T_START_VERBOSE(p->t_coadd);
	f_coadd(image, p->binned_frame);
	T_STOP_VERBOSE(p->t_coadd);
}

/**
 * \brief Entry function for Benchmark #1.1: Image corrections and calibrations.
 */
void benchmark_1_1_image_preproc(benchmark_params_t *p)
{
	unsigned int frame_i;
	
	/* Loop through each frames and perform pre-processing. */
	T_START(p->t_test);
	for(frame_i=0; frame_i<num_frames; frame_i++)
	{
		T_START(p->t_frame[frame_i]);
		benchmark_1_1_proc_frame(p, &frame[frame_i], frame_i);
		T_STOP(p->t_frame[frame_i]);

		// FIXME radiation correction frame handling to be implemented

	}
	T_STOP(p->t_test);
}

float calc_avg_t(time_t *ts, unsigned int length)
{
	unsigned int i;
	float sum = 0; 

	for(i=0; i<length; i++)
	{
		sum += ts[i];
	}

	return sum / (float)length;
}

void benchmark_1_1_print_results(benchmark_params_t *params)
{
	unsigned int frame_i;
	float frame_avg_t, frame_max_t, frame_min_t;

	/* Calculate average times */
	frame_avg_t = calc_avg_t(p->t_frames, p->num_frames);

	/* Calculate max/min times */
	frame_max_t = p->t_frame[0];
	frame_min_t = p->t_frame[0];
	for(frame_i=1; frame_i<p->num_frames; frame_i++)
	{
		if(frame_max_t < p->t_frame[frame_i]) {
			frame_max_t = p->t_frame[frame_i];
		}

		if(frame_min_t > p->t_frame[frame_i]) {
			frame_min_t = p->t_frame[frame_i];
		}
	}

	/* Print output */
	printf(
			"Results:\n"
			"\tTotal test time: %.3f\n"
			"\tAverage time per frame: %.3f\n"
			"\tMax frame time: %.3f\n"
			"\tMin frame time: %.3f\n"
			,
			p->t_test,
			frame_avg_t,
			frame_max_t,
			frame_min_t
		); 


#ifdef OBPMARK_VERBOSE_TIMING
	// FIXME add verbose timing info printing
#endif // OBPMARK_VERBOSE_TIMING
}

int benchmark_1_1_log_results(benchmark_params_t *params)
{
	// FIXME implement
	return 1;
}

int benchmark_1_1(unsigned int frame_width, unsigned int frame_height)
{
	benchmark_params_t params; 

	printf("OBPMark Benchmark #1.1: Image corrections and calibrations.\n");
	
	/* Allocate buffers */
	printf("Allocating buffers...\n");
	if(!benchmark_1_1_alloc_buffers(&params), frame_width, frame_height) {
		printf("fatal error: could not allocate buffers for benchmark.\n");
		return 1;
	}
	
	/* Load calibration data */
	printf("Loading calibration data...\n");
	if(!benchmark_1_1_load_calibration_data(&params)) {
		printf("fatal error: could not load calibration data.\n");
		benchmark_1_1_cleanup(&params);
		return 2;
	}

	/* Run benchmarks */
	printf("Running benchmark...\n");
	benchmark_1_1_image_preproc(&params);

	/* Print results report */
	printf("Benchmark completed successfully.\n");
	benchmark_1_1_print_results(&params); 

	/* Log results report */
	printf("Logging results.\n");
	if(!benchmark_1_1_log_results(&params)) {
		printf("fatal error: could not save results log.\n");
		benchmark_1_1_cleanup(&params);
		return 3;
	}

	benchmark_1_1_cleanup(&params);
	printf("Finished successfully.\n");
	return 0; 
}

int main()
{
	int ret; 
	/* Test ID: #1: Frame size: 1024 x 1024 */
	ret = benchmark_1_1(1024, 1024);
       	if(ret != 0) {
		printf("Test ID #1 failed.\n");
		return ret;
	}

	/* Test ID: #2: Frame size: 2048 x 2048 */
	ret = benchmark_1_1(2048, 2048);
       	if(ret != 0) {
		printf("Test ID #2 failed.\n");
		return ret;
	}

	/* Test ID: #3: Frame size: 4096 x 4096 */
	ret = benchmark_1_1(4096, 4096);
       	if(ret != 0) {
		printf("Test ID #3 failed.\n");
		return ret;
	}

	return 0;
}

