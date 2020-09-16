# OBPMark - Data Generation for Benchmark #1.1: Image Calibrations and Corrections
# Author: david.steenari@esa.int
#
# European Space Agency Community License V2.3 applies.
# For more info see the LICENSE file in the root folder.

################################## SETTINGS ####################################
CC = gcc
CFLAGS = -g -O3 -I$(COMMON_PATH)

################################################################################
COMMON_PATH = ../../../src/common

_COMMON_INCLUDES = image_util.h image_mem_util.h image_file_util.h 
_COMMON_SOURCES  = image_mem_util.c image_file_util.c
COMMON_OBJECTS   = image_mem_util.o image_file_util.o

COMMON_INCLUDES = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_INCLUDES))
COMMON_SOURCES  = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_SOURCES))

#_INCLUDES = 
#INCLUDES = $(pathsubst %,$(INCLUDE_PATH)/%,$(_INCLUDES)) $(COMMON_INCLUDES)
INCLUDES = $(COMMON_INCLUDES)

OBJECTS = datagen-1.1-image.o $(COMMON_OBJECTS)
OUTFILE = datagen-1.1-image


################################################################################

datagen-1.1-image: $(OBJECTS)
	$(CC) $(CFLAGS) -o $(OUTFILE) $(OBJECTS)

image_mem_util.o: $(COMMON_PATH)/image_mem_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_mem_util.c

image_file_util.o: $(COMMON_PATH)/image_file_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_file_util.c

timing.o: $(COMMON_PATH)/timing.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/timing.c

datagen-1.1-image.o: datagen-1.1-image.c $(INCLUDES)
	$(CC) $(CFLAGS) -c datagen-1.1-image.c

clean:
	rm $(OUTFILE) $(OBJECTS)
