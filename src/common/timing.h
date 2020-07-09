/**
 * \brief OBPMark common functions for timing
 * \file timing.h
 * \author David Steenari
 * 
 * To enable timing measurements, define "OBPMARK_TIMING 1". 
 * To enable verbose timing measurements, define "OBPMARK_TIMING 2". 
 */
#ifndef OBPMARK_TIMING_H_
#define OBPMARK_TIMING_H_

/* Stdlib implemenation, replace if other timing library is to be used */
#if (OBPMARK_TIMING > 0)
#include <time.h>
#define T_INIT(t_timer) time_t t_timer
#define T_START(t_timer) t_timer = clock()
#define T_STOP(t_timer) t_timer = (clock() - t_timer)
#define T_TO_SEC(t_timer) ((float)t_timer/CLOCKS_PER_SEC)
#else
#define T_START(t_timer)
#define T_STOP(t_timer)
#endif

#endif // OBPMARK_TIMING_H_
