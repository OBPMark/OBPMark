# OBPMark top-level Makefile 
# For license information, see the LICENSE file in the top level folder. 
# Author: David Steenari (ESA)

# Settings 
include config/all.config

# Benchmarks to build 
SUBDIRS := 1.1-image
#SUBDIRS += 1.2-radar
SUBDIRS += 2.1-data_compression
SUBDIRS += 2.2-image_compression
#SUBDIRS += 2.3-hyperspectral_compression
#SUBDIRS += 3.1-aes_compression
SUBDIRS += 4.1-fir_filter
SUBDIRS += 4.2-fft
SUBDIRS += 4.3-convolution
SUBDIRS += 4.4-matrix_multiplication
#SUBDIRS += 5.1-object_detection
#SUBDIRS += 5.2-cloud_screening

# Targets

.phony: all 
all: cpu

cpu: 
	for dir in $(SUBDIRS); do \
		$(MAKE) cpu -C src/$$dir; \
	done

openmp:
	for dir in $(SUBDIRS); do \
		$(MAKE) openmp -C src/$$dir; \
	done

opencl:
	for dir in $(SUBDIRS); do \
		$(MAKE) opencl -C src/$$dir; \
	done

cuda:
	for dir in $(SUBDIRS); do \
		$(MAKE) cuda -C src/$$dir; \
	done

hip:
	for dir in $(SUBDIRS); do \
		$(MAKE) hip -C src/$$dir; \
	done

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) clean -C src/$$dir; \
	done
	rm -rf ./build
	rm -rf ./bin

