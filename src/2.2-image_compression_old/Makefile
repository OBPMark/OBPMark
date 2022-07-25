# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
# the build target executable:
TARGET = ccds122
# FLAGS 
# CC compiler flags:
CFLAGS  = -g
# NVCC compiler flags
NVCCFLAGS = -arch compute_75 -code sm_75
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL  
# OPENMP FLAGS
OMPFLAGS = -fopenmp -lm


# FOLDERS
# CPU FOLDER
CPUFOLDER = ./cpulib/
# CUDA FOLDER
CUFOLDER = ./cudalib/
# OPENCL FOLDER
OPFOLDER = ./opencllib/
# OPENMP FOLDER
OMPFOLDER = ./openmplib/
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/



# COMPILER MAIN


# By defult compiles the CPU version
all: lib_cpu.o easyBMP/EasyBMP.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp $(CPUFOLDER)lib_cpu.o -o $(OUTPUTFOLDER)$(TARGET) easyBMP/EasyBMP.o  $(CFLAGS)

lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CC)  -c $(CPUFOLDER)lib_cpu.cpp   -o $(CPUFOLDER)lib_cpu.o  $(CFLAGS)

# common
EasyBMP.o: easyBMP/EasyBMP.cpp easyBMP/EasyBMP*.h
	g++ $(CFLAGSBMP) -c easyBMP/EasyBMP.cpp

###############################################################################################################################################
# CUDA part
###############################################################################################################################################
.PHONY: cuda
cuda: main_cuda

main_cuda: lib_cuda.o easyBMP/EasyBMP.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp -DCUDA $(CUFOLDER)lib_cuda.o -o $(OUTPUTFOLDER)$(TARGET)  easyBMP/EasyBMP.o $(CUFLAGS) $(CFLAGS) -lstdc++

lib_cuda.o: $(CUFOLDER)lib_cuda.cu
	$(NVCC)  -c $(CUFOLDER)lib_cuda.cu  -DCUDA -o $(CUFOLDER)lib_cuda.o $(NVCCFLAGS)


###############################################################################################################################################
# OpenCL part
###############################################################################################################################################
.PHONY: opencl
opencl: main_opencl

main_opencl:lib_opencl.o easyBMP/EasyBMP.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) main.cpp -DOPENCL $(OPFOLDER)lib_opencl.o -o $(OUTPUTFOLDER)$(TARGET)  easyBMP/EasyBMP.o $(OPFLAGS) $(CFLAGS) -lstdc++
lib_opencl.o:
	/bin/bash CLHT.sh 
	$(CC)  -c $(OPFOLDER)lib_opencl.cpp  -DOPENCL -o $(OPFOLDER)lib_opencl.o $(OPFLAGS)

###############################################################################################################################################
# OpemMP part
###############################################################################################################################################
.PHONY: openmp
openmp:  main_openmp

main_openmp: lib_omp.o easyBMP/EasyBMP.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) main.cpp -DOPENMP $(OMPFOLDER)lib_omp.o -o $(OUTPUTFOLDER)$(TARGET) easyBMP/EasyBMP.o $(CFLAGS) $(OMPFLAGS)

lib_omp.o: $(OMPFOLDER)lib_omp.cpp 
	export OMP_NUM_THREADS=8
	$(CC) -c $(OMPFOLDER)lib_omp.cpp -DOPENMP -o $(OMPFOLDER)lib_omp.o $(CFLAGS) $(OMPFLAGS)

###############################################################################################################################################
# END CUDA part
###############################################################################################################################################
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)
	