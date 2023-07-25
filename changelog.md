## OBPMark Changelog
[2023-07-25] v0.3.1 release
- Overall:
	- Change config files to add -O3 for optimizations as well to remove debug flags.
	- Change in 2.2 Benchmark to display the DWT execution time and the BPE execution time.
	- Change in the calculations of throughput in benchmarks 1.1 and 1.2 to measure the computation throughput and not the output throughput.
	- Minor fixes to remove Warnings in compilation.


[2022-08-08] v0.3 release
- Benchmark #1.1 "Image Correction and Calibration"
	- OpenCL bug fixes.

- Added Benchmark #1.2: Radar Image Processing.

- Benchmark #2.1 "CCSDS 121.0"
	- Updated verified version of benchmark with several bug fixes.

- Adeed Benchmark #2.2: CCSDS 122.0 Image Compression.

- Overall:
	- Change standard output of the performance of the benchmarks. 
	- Minor bugfixes.


[2022-05-07] v0.2 release
- Benchmark #1.1 "Image Correction and Calibration"
	- Replaced input and verification data
	- Updated radiation scrubbing function 
	
- Benchmark #2.1 "CCSDS 121.0"
	- Added input and verification data
	- Fixed bugs in implement 
	
- Added Benchmark #3.1 "AES Encryption" 
	- New reference implementations provided
	 
- Replaced Benchmark #4 with signal processing applications (previous building block benchmarks moved to dedicated repository: OBPMark-Kernel / GPU4S_Bench)
- Moved Benchmarks #5.1 and #5.2 to separate benchmark repository for machine learning: OBPMark-ML
- Overall: 
	- Consolidated benchmark code style 
	- Updated build scripts

[2021-06-14] v0.1 release
- Public release of existing benchmarks
- Benchmark #1.1 "Image Correction and Calibration"
- Benchmark #2.1 "CCSDS 121.0"
- Benchmark #2.2 "CCSDS 122.0" 
- Processing building block benchmarks
