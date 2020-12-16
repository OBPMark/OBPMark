# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
HIP = /opt/rocm/hip/bin/hipcc
# the build target executable:
TARGET = obpmark_image
# FLAGS 
# PRINT flag
PRINT = #-DPRINT
# CC compiler flags:
CFLAGS  = -g
# NVCC compiler flags
NVCCFLAGS = -arch compute_75 -code sm_75
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS =  -lOpenCL 
# OPENMP FLAGS
OMPFLAGS = -fopenmp -lm
# HIP FLAGS
HIPFLAGS = -I/opt/rocm/hip/include -L/opt/rocm/hip/lib

# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 

# FOLDERS
# CUDA FOLDER
CUFOLDER = ./cuda/
# OPENCL FOLDER
OPFOLDER = ./opencl/
# OPENMP FOLDER
OMPFOLDER = ./openmp/
# HIP FOLDER
HIPFOLDER = ./hip/
# CPU FOLDER
CPUFOLDER = ./cpu/
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

# COMPILER MAIN
all:
	@echo "YOU NEED TO SELECT CUDA OR OPENCL -> make CUDA / make OpenCL / make OpenMP / make Hip / make Sequential"
# End Main
# Shortcuts
.PHONY: all-bin
all-bin: cuda opencl openmp hip 
.PHONY: all-cuda
all-cuda: cuda 
.PHONY: all-opencl
all-opencl: opencl 
.PHONY: all-openmp
all-openmp: openmp 
.PHONY: all-hip
all-hip: hip 
.PHONY: CUDA
CUDA: cuda
.PHONY: OpenCL
OpenCL: opencl
.PHONY: OpenMP
OpenMP: openmp
.PHONY: Hip
Hip: hip
.PHONY: Sequential
Sequential: sequential
# End Shortcuts

# CUDA part
.PHONY: cuda
cuda: main_cuda

cuda_processing.o: $(CUFOLDER)processing.cu
	$(NVCC) -DCUDA -c $(CUFOLDER)processing.cu  -o $(CUFOLDER)cuda_processing.o $(NVCCFLAGS)

main_cuda: cuda_processing.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp  -DCUDA $(CUFOLDER)cuda_processing.o -o $(OUTPUTFOLDER)$(TARGET)_cuda $(CUFLAGS) $(CFLAGS) -lstdc++

# End CUDA

# OpenCL Part
opencl:  main_opencl

opencl_processing.o: $(OPFOLDER)processing.c
	/bin/bash CLHT.sh
	$(CC) -DOPENCL -c $(OPFOLDER)processing.c  -o $(OPFOLDER)opencl_processing.o $(OPFLAGS)

main_opencl: opencl_processing.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp  -DOPENCL $(OPFOLDER)opencl_processing.o -o $(OUTPUTFOLDER)$(TARGET)_opencl $(OPFLAGS) $(CFLAGS) -lstdc++

# End OpenCL

# OpenMP Part
openmp:  main_openmp

openmp_processing.o: $(OMPFOLDER)processing.c
	export OMP_NUM_THREADS=8
	$(CC) -DOPENMP -c $(OMPFOLDER)processing.c  -o $(OMPFOLDER)openmp_processing.o $(OMPFLAGS)

main_openmp: openmp_processing.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp  -DOPENMP $(OMPFOLDER)openmp_processing.o -o $(OUTPUTFOLDER)$(TARGET)_openmp $(OMPFLAGS) $(CFLAGS) -lstdc++
# End OpenMP

# Hip part
hip: main_hip

lib_hip.o: $(HIPFOLDER)lib_hip.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip.cpp -o $(HIPFOLDER)lib_hip.o $(CFLAGS) $(HIPFLAGS)

main_hip: main.cpp lib_hip.o lib_cpu.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp $(HIPFOLDER)lib_hip.o  $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET)_hip_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

# CPU part
sequential: main_cpu


cpu_processing.o: $(CPUFOLDER)processing.c
	$(CC) -c $(CPUFOLDER)processing.c  -o $(CPUFOLDER)cpu_processing.o 

main_cpu: cpu_processing.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp  $(CPUFOLDER)cpu_processing.o -o $(OUTPUTFOLDER)$(TARGET)_sequential  $(CFLAGS) -lstdc++

# End CPU 

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf *.out
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(OMPFOLDER)*.o
	rm -rf $(HIPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_*
	rm -rf $(OUTPUTFOLDER)*.out
