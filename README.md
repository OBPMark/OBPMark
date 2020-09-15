### OBPMark (On Board Processing Benchmark)
Author: David Steenari (david.steenari@esa.int)

This repository contains a set of reference implementations for performing benchmarks on devices and systems on-board spacecraft. 

The following folders are in the repository: 

	docs/	The TeX files for building the OBPMark specification. 
	src/	Source files for each of the benchmarks. 

The benchmarks are organised in the following structure: 

	src/1.1-image/
	src/1.2-radar/
	src/2.1-data_compression/
	src/2.2-image_compression/
	src/2.3-hyperspectral_compression/
	src/3.1-aes_compression/
	src/4.1-fir_filter/
	src/4.2-fft/
	src/5.1-object_detection/
	src/5.2-cloud_screening/
	src/common/

The entire set of benchmarks can be built by invoking the Makefile in the top src/ directory, or by invoking the individual Makefiles in each of the src sub-directories. 


