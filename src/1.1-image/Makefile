# OBPMark Benchmark #1.1 "Image processing" Makefile
# Author: David Steenari (ESA)
include ../../config/all.config

################################# CONFIGURUATION ##############################
EXEC_NAME	:= obpmark-1.1-image

# Relative paths
CONFIG_PATH	:= ../../$(CONFIG_FOLDER)
BUILD_PATH 	:= ../../$(BUILD_FOLDER)/1.1-image
BIN_PATH 	:= ../../$(BIN_FOLDER)/1.1-image
EXEC_PATH	:= $(BIN_PATH)/$(EXEC_NAME)

# - Sequential - 
SEQ_TARGET	:= $(BUILD_PATH)/$(SEQ_FOLDER)
SEQ_EXEC	:= $(EXEC_PATH)_$(SEQ_NAME)

# - OpenMP -
OPENMP_TARGET	:= $(BUILD_PATH)/$(OPENMP_FOLDER)
OPENMP_EXEC	:= $(EXEC_PATH)_$(OPENMP_NAME)

# - OpenCL - 
OPENCL_TARGET	:= $(BUILD_PATH)/$(OPENCL_FOLDER)
OPENCL_EXEC	:= $(EXEC_PATH)_$(OPENCL_NAME)

# - CUDA - 
CUDA_TARGET	:= $(BUILD_PATH)/$(CUDA_FOLDER)
CUDA_EXEC	:= $(EXEC_PATH)_$(CUDA_NAME)

# - HIP - 
HIP_TARGET	:= $(BUILD_PATH)/$(HIP_FOLDER)
HIP_EXEC	:= $(EXEC_PATH)_$(HIP_NAME)

############################## LOAD EXTERNAL CONFIG ############################

ifeq ($(MAKECMDGOALS),cpu)
include $(CONFIG_PATH)/$(SEQ_CONFIG)
endif

ifeq ($(MAKECMDGOALS),openmp)
include $(CONFIG_PATH)/$(OPENMP_CONFIG)
endif

ifeq ($(MAKECMDGOALS),opencl)
include $(CONFIG_PATH)/$(OPENCL_CONFIG)
endif

ifeq ($(MAKECMDGOALS),cuda)
include $(CONFIG_PATH)/$(CUDA_CONFIG)
endif

ifeq ($(MAKECMDGOALS),hip)
include $(CONFIG_PATH)/$(HIP_CONFIG)
endif

################################# COMPILER MAIN ###############################
all:  
	@echo "error: must specify target: cpu, openmp, opencl, cuda, hip"

################################### SEQUENTIAL ################################
# Shortcuts
.PHONY: CPU
CPU: cpu

# Build target
cpu: main_cpu

cpu_conf:
	mkdir -p $(SEQ_TARGET)

cpu_processing.o: cpu_conf $(SEQ_SRC)/processing.c
	$(CC) $(CFLAGS) -c $(SEQ_SRC)/processing.c -o $(SEQ_TARGET)/cpu_processing.o

main_cpu: cpu_conf cpu_processing.o
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(SRC_FOLDER)/main.cpp $(SEQ_TARGET)/cpu_processing.o -o $(SEQ_EXEC)

##################################### OPENMP #################################
# Shortcuts
.PHONY: all-openmp
all-openmp: openmp 
.PHONY: OpenMP
OpenMP: openmp

# Build target
openmp: main_openmp

openmp_conf:
	mkdir -p $(OPENMP_TARGET)
	export OMP_NUM_THREADS # FIXME this could probably be deleted, test first 

openmp_processing.o: openmp_conf $(OPENMP_SRC)/processing.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOPENMP -c $(OPENMP_SRC)/processing.c  -o $(OPENMP_TARGET)/openmp_processing.o # FIXME -DOPENMP should be variable above, check others below as well

main_openmp: openmp_conf openmp_processing.o
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_FOLDER)/main.cpp  -DOPENMP $(OPENMP_TARGET)/openmp_processing.o -o $(OPENMP_EXEC)

##################################### OPENCL #################################
# Shortcuts
.PHONY: all-opencl
all-opencl: opencl 
.PHONY: OpenCL
OpenCL: opencl

# Build target
opencl: main_opencl

opencl_conf:
	mkdir -p $(OPENCL_TARGET)

opencl_processing.o: $(OPENCL_SRC)/processing.c
	/bin/bash $(CONFIG_PATH)/CLHT.sh
	$(CC) $(OPFLAGS) -DOPENCL -c $(OPENCL_SRC)/processing.c  -o $(OPENCL_TARGET)/opencl_processing.o

main_opencl: opencl_processing.o
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(OPFLAGS) $(SRC_FOLDER)/main.cpp  -DOPENCL $(OPENCL_TARGET)/opencl_processing.o -o $(OPENCL_EXEC)

###################################### CUDA ##################################
# Shortcuts
.PHONY: all-cuda
all-cuda: cuda 
.PHONY: CUDA
CUDA: cuda

# Build target
cuda: main_cuda

cuda_conf:
	mkdir -p $(CUDA_TARGET)

cuda_processing.o: cuda_conf $(CUDA_SRC)/processing.cu
	$(NVCC) $(NVCCFLAGS) -DCUDA -c $(CUDA_SRC)/processing.cu  -o $(CUDA_TARGET)/cuda_processing.o

main_cuda: cuda_conf cuda_processing.o
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(CUFLAGS) $(SRC_FOLDER)/main.cpp -DCUDA $(CUDA_TARGET)/cuda_processing.o -o $(CUDA_EXEC)

###################################### HIP ###################################
# Shortcuts
.PHONY: all-hip
all-hip: hip 
.PHONY: HIP
HIP: hip
.PHONY: Hip
Hip: hip

# Build target
hip: main_hip

hip_conf:
	mkdir -p $(HIP_TARGET)

lib_hip.o: hip_conf $(HIP_SRC)/lib_hip.cpp
	$(HIP) $(CFLAGS) $(HIPFLAGS) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIP_SRC)/lib_hip.cpp -o $(HIP_TARGET)/lib_hip.o

main_hip: hip_conf main.cpp lib_hip.o lib_cpu.o
	mkdir -p $(BIN_PATH) 
	$(HIP) $(CFLAGS) $(HIPFLAGS) -D$(DATATYPE) -DHIP $(SRC_FOLDER)/main.cpp $(BUILD_PATH)/lib_hip.o  $(SEQ_FOLDER)/lib_cpu.o -o $(HIP_EXEC)_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) 

##################################### CLEAN ##################################
.PHONY: clean
clean:
	rm -f $(SEQ_TARGET)/*.o
	rm -f $(OPENMP_TARGET)/*.o
	rm -f $(OPENCL_TARGET)/*.o
	rm -f $(CUDA_TARGET)/*.o
	rm -f $(HIP_TARGET)/*.o
	rm -f $(EXEC_PATH)_*
