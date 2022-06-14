
NVCC = /usr/local/cuda/bin/nvcc
CC = gcc
CXX = g++

SRC_DIR := src
CPU_OBJ_DIR := obj_cpu
OMP_OBJ_DIR := obj_omp
CUDA_OBJ_DIR := obj_cuda
BIN_DIR := bin
CPU_DIR := $(SRC_DIR)/cpulib
CUDA_DIR := $(SRC_DIR)/cudalib
OPENMP_DIR := $(SRC_DIR)/openmplib
OPENCL_DIR := $(SRC_DIR)/opencllib

# Do not modify
SRC := $(wildcard $(SRC_DIR)/*.c)
EXE := $(BIN_DIR)/CCSDS121
CPPFLAGS := -Iinclude -MMD -MP
NVCCPPFLAGS := -Iinclude
CFLAGS   := -Wall
LDFLAGS  := -Llib
LDLIBS   := -lm
OMPFLAGS := -fopenmp -DOPENMP
CUFLAGS  := -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart
NVCCFLAGS := -arch compute_75 -code sm_75
OPFLAGS := -I/usr/local/cuda/include/   -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL  
.PHONY: all openmp clean

# Makes all the available targets
all: main_cpu

######################### CPU #########################


	@echo "-- CPU version --"

SRC_CPU := $(wildcard $(CPU_DIR)/*.c)
OBJ_CPU := $(SRC_CPU:$(CPU_DIR)/%.c=$(CPU_OBJ_DIR)/%.o)
CPU_MAIN_OBJ := $(SRC:$(SRC_DIR)/%.c=$(CPU_OBJ_DIR)/%.o)


main_cpu: $(OBJ_CPU) $(CPU_MAIN_OBJ) | $(BIN_DIR)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $(EXE)

$(CPU_OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(CPU_OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(CPU_OBJ_DIR)/%.o: $(CPU_DIR)/%.c | $(CPU_OBJ_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@


######################### OPENMP #########################


.PHONY: openmp
openmp: main_openmp

	@echo "-- OpenMP version --"

SRC_OMP := $(wildcard $(OPENMP_DIR)/*.c)
OBJ_OMP := $(SRC_OMP:$(OPENMP_DIR)/%.c=$(OMP_OBJ_DIR)/%.o)
OMP_MAIN_OBJ := $(SRC:$(SRC_DIR)/%.c=$(OMP_OBJ_DIR)/%.o)


main_openmp: $(OBJ_OMP) $(OMP_MAIN_OBJ) | $(BIN_DIR)
	$(CC) $(OMPFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $(EXE)

$(OMP_OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OMP_OBJ_DIR)
	$(CC) $(CPPFLAGS) $(OMPFLAGS) $(CFLAGS) -c $< -o $@

$(OMP_OBJ_DIR)/%.o: $(OPENMP_DIR)/%.c | $(OMP_OBJ_DIR)
	$(CC) $(CPPFLAGS) $(OMPFLAGS) $(CFLAGS) -c $< -o $@


######################### CUDA #########################
.PHONY: cuda
cuda: main_cuda

	@echo "-- CUDA version --"


SRC_CUDA := $(wildcard $(CUDA_DIR)/*.c)
OBJ_CUDA := $(SRC_CUDA:$(CUDA_DIR)/%.cu=$(CUDA_OBJ_DIR)/%.o)
CUDA_MAIN_OBJ := $(SRC:$(SRC_DIR)/%.c=$(CUDA_OBJ_DIR)/%.o)


#main_cuda: $(OBJ_CUDA) $(CUDA_MAIN_OBJ) | $(BIN_DIR)
#	$(CC) $(CUFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $(EXE)

#$(CUDA_OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(CUDA_OBJ_DIR)
#	$(CC) $(NVCCPPFLAGS) $(CUFLAGS)  -c $< -o $@

#$(CUDA_OBJ_DIR)/%.o: $(CUDA_DIR)/%.cu | $(CUDA_OBJ_DIR)
#	$(NVCC)  $(NVCCPPFLAGS) $(CUFLAGS) -c $< -o $@

main_cuda: lib_cuda.o 
	mkdir -p $(BIN_DIR) 
	$(CXX)  $(NVCCPPFLAGS) $(SRC_DIR)/CCSDS121.c -DCUDA $(CUDA_DIR)/lib_cuda.o -o $(EXE)  $(CUFLAGS) $(CFLAGS) -lstdc++

lib_cuda.o: $(CUDA_DIR)/lib_cuda.cu
	$(NVCC)  -c $(CUDA_DIR)/lib_cuda.cu  $(NVCCPPFLAGS) -DCUDA -o $(CUDA_DIR)/lib_cuda.o $(NVCCFLAGS)


######################### OPENCL #########################
.PHONY: opencl
opencl: main_opencl

	@echo "-- OpenCL version --"

main_opencl: lib_opencl.o 
	mkdir -p $(BIN_DIR) 
	$(CXX)  $(NVCCPPFLAGS) $(SRC_DIR)/CCSDS121.c  $(OPENCL_DIR)/lib_opencl.o -o $(EXE)   $(OPFLAGS)  $(CFLAGS)  -lstdc++ -DOPENCL

lib_opencl.o: $(OPENCL_DIR)/lib_opencl.cpp
	/bin/bash $(SRC_DIR)/CLHT.sh 
	$(CXX)  -c $(OPENCL_DIR)/lib_opencl.cpp  -o $(OPENCL_DIR)/lib_opencl.o  $(NVCCPPFLAGS)  $(OPFLAGS) -DOPENCL
 

######################### GENERIC RULES #########################


$(BIN_DIR) $(OMP_OBJ_DIR) $(CPU_OBJ_DIR) $(CUDA_OBJ_DIR):
	mkdir -p $@


######################### CLEAN #########################

clean:
	@$(RM) -rv $(BIN_DIR) $(OMP_OBJ_DIR) $(CPU_OBJ_DIR) $(CUDA_OBJ_DIR)

-include $(OBJ:.o=.d)