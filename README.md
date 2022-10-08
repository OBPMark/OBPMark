### OBPMark (On Board Processing Benchmark)
Current version: v0.3

Contact: OBPMark@esa.int
Website: OBPMark.org

## What is OBPMark?
OBPMark (On-Board Processing Benchmarks) is a set of computational performance benchmarks developed specifically for spacecraft on-board data processing applications, such as: image and radar processing, data and image compressions, signal processing and machine learning.

The development of OBPMark was driven by the lack of openly available and representative benchmarks for space applications.

OBPMark consists of three main components:

- A technical note describing the benchmarks, their implementation and result reporting.
- Reference implementations in C (sequential) and standard parallelisation schemes, such as OpenMP, OpenCL and CUDA.
- A list of known published performance benchmark results.

OBPMark has been developed in cooperation by the European Space Agency (ESA) and Barcelona Supercomputing Center (BSC).
The OBPMark definition and implementation has been made available partially through funding from the ESA “General Studies Programme (GSP)”.

OBPMark is openly available, and contributions from the community are warmly welcome.

## Public Beta Version Notice
Please note that OBPMark is currently in "Public Beta" while the implementations and verifications of the benchmarks are being completed.

Expect features and data to be missing. For an overview of the main current issues and missing features, see the "Issues" tab at the top. 

In the meanwhile, users are invited to test out the existing implementations (see list below) and report bugs and issues via the github interface.

## Changelog
See separate file for [changelog.md](changelog.md)

## Documentation
Detailed technical description of the benchmarks are included in the OBPMark Technical Note. 

See Git wiki page for documentation on the software implementations: https://github.com/OBPMark/OBPMark/wiki/
 
## Compilation and Usage

Documentation regarding how to compile and use the software is located in the Git wiki page: https://github.com/OBPMark/OBPMark/wiki/User-Instructions

# Contributors 
Authors/Chairs:  
- David Steenari, European Space Agency (ESA)  
- Dr. Leonidas Kosmidis, Barcelona Supercomputing Center (BSC)  
  
Contributors:  
- Alvaro Jover-Alvarez, Universitat Politècnica de Catalunya and Barcelona Supercomputing Center (UPC/BSC):
	- Implementation of OpenMP versions. 
- Ivan Rodriguez Ferrandez, Universitat Politècnica de Catalunya and Barcelona Supercomputing Center (UPC/BSC):
	- Implementation of Benchmark #1.1 "Image Calibration and Correction"
	- Implementation of Benchmark #2.1 "CCSDS 121.0 Data Compression"
	- Implementation of Benchmark #2.1 "CCSDS 122.0 Image Compression"
	- Verification of implementations.  
- Marc Solé Bonet, Barcelona Supercomputing Centrer (BSC):
	- Implementation of Benchmark #3.1 "AES Encryption"

## Description
This repository contains a set of reference implementations for performing benchmarks on devices and systems on-board spacecraft. 

The following folders are in the repository: 

	config/ Contains build configuration files for all benchmarks compliation formats (C, OpenMP, OpenCL, CUDA). Can be customized if needed.
	data/	Contains input and output/verification data that is to be used during implementation and benchmark exeuction.
	docs/	The OBPMark technical documentation.
	src/	Source files for each of the benchmarks. 
	tools/	Contains helper/support tools that are not required for the benchmarking. 

The benchmarks are organised in the following structure ("Public Beta" current status also shown): 

	src/1.1-image/				-- Available in C (sequential), OpenMP, OpenCL and CUDA.
	src/1.2-radar/				-- Currently under development, see "radar-dev" branch.
	src/2.1-data_compression/		-- Available in C (sequential), OpenMP, OpenCL and CUDA.
	src/2.2-image_compression/		-- Currently under development, see "ccsds122-dev" branch.
	src/2.3-hyperspectral_compression/	-- TBA.
	src/3.1-aes_encryption/			-- Available in C (sequential), OpenMP, OpenCL and CUDA.
	src/4.1-dvbs2x_modulation/		-- TBA.
	src/4.2-dvbs2x_demodulation/		-- TBA.
	src/common/

The entire set of benchmarks can be built by invoking the Makefile in the top src/ directory, or by invoking the individual Makefiles in each of the src sub-directories. 

