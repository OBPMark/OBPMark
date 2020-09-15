# OBPMark - Benchmark #1.1: Image Calibrations and Corrections
# Author: david.steenari@esa.int
#
# European Space Agency Community License V2.3 applies.
# For more info see the LICENSE file in the root folder.

################################## SETTINGS ####################################
# FIXME add build options for OpenMP, OpenCL and CUDA
INCLUDE_PATH = ./sequential
SOURCE_PATH = ./sequential

CC = gcc
CFLAGS = -g -O3 -I$(INCLUDE_PATH)

################################################################################
COMMON_PATH = ../common

_COMMON_INCLUDES = timing.h image_util.h image_mem_util.h image_file_util.h 
_COMMON_SOURCES  = image_mem_util.c image_file_util.c
COMMON_OBJECTS   = image_mem_util.o image_file_util.o

COMMON_INCLUDES = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_INCLUDES))
COMMON_SOURCES  = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_SOURCES))

_INCLUDES = image_kernels.h
INCLUDES = $(pathsubst %,$(INCLUDE_PATH)/%,$(_INCLUDES)) $(COMMON_INCLUDES)

OBJECTS = image_kernels.o $(COMMON_OBJECTS)
OUTFILE = obpmark_1_1-image


################################################################################

obpmark1_1-image: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(OUTFILE) $(OBJECTS)

image_mem_util.o: $(COMMON_PATH)/image_mem_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_mem_util.c

image_file_util.o: $(COMMON_PATH)/image_file_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_file_util.c

timing.o: $(COMMON_PATH)/timing.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/timing.c

image_kernels.o: $(SOURCE_PATH)/image_kernels.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(SOURCE_PATH)/image_kernels.c

benchmark1_1.o: benchmark1_1.c $(INCLUDES)
	$(CC) $(CFLAGS) -c benchmark1_1.c

clean:
	rm $(OUTFILE) $(OBJECTS)
