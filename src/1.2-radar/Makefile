# OBPMark Benchmark #1.2 "Image processing" Makefile
# Author: David Steenari (ESA)
include ../../config/all.config

################################# CONFIGURUATION ##############################
EXEC_NAME	:= obpmark-1.2-radar

# Relative paths
CONFIG_PATH	:= ../../$(CONFIG_FOLDER)
BUILD_PATH 	:= ../../$(BUILD_FOLDER)/1.2-radar
BIN_PATH 	:= ../../$(BIN_FOLDER)/1.2-radar
EXEC_PATH	:= $(BIN_PATH)/$(EXEC_NAME)

################################################################################
COMMON_PATH = ../../src/common

_COMMON_INCLUDES = image_util.h image_mem_util.h image_file_util.h 
_COMMON_SOURCES  = image_mem_util.c image_file_util.c util_data_rand.c util_arg.c util_data_files.c 
COMMON_OBJECTS   = image_mem_util.o image_file_util.o util_data_rand.o util_arg.o util_data_files.o 
COMMON_OBJECTS_PATH = $(COM_TARGET)/image_mem_util.o $(COM_TARGET)/image_file_util.o $(COM_TARGET)/util_data_rand.o $(COM_TARGET)/util_arg.o $(COM_TARGET)/util_data_files.o

COMMON_INCLUDES = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_INCLUDES))
COMMON_SOURCES  = $(pathsubst %,$(COMMON_PATH)/%,$(_COMMON_SOURCES))
################################################################################

# - Common -
COM_TARGET	:= $(BUILD_PATH)/$(COMM_FOLDER)

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
all: cpu openmp opencl cuda #hip
#	@echo "error: must specify target: cpu, openmp, opencl, cuda, hip"

################################### COMMON ################################
common_conf:
	mkdir -p $(COM_TARGET)

image_mem_util.o: common_conf $(COMMON_PATH)/image_mem_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_mem_util.c -o $(COM_TARGET)/image_mem_util.o

image_file_util.o: common_conf $(COMMON_PATH)/image_file_util.c $(INCLUDES)
	$(CC) $(CFLAGS) -c $(COMMON_PATH)/image_file_util.c -o $(COM_TARGET)/image_file_util.o

util_data_rand.o: common_conf $(SRC_FOLDER)/util_data_rand.c
	$(CC) $(CFLAGS) -c $(SRC_FOLDER)/util_data_rand.c -o $(COM_TARGET)/util_data_rand.o

util_arg.o: common_conf $(SRC_FOLDER)/util_arg.c
	$(CC) $(CFLAGS) -c $(SRC_FOLDER)/util_arg.c -o $(COM_TARGET)/util_arg.o

util_data_files.o: common_conf $(SRC_FOLDER)/util_data_files.c
	$(CC) $(CFLAGS) -c $(SRC_FOLDER)/util_data_files.c -o $(COM_TARGET)/util_data_files.o

################################### SEQUENTIAL ################################
# Shortcuts
.PHONY: CPU
CPU: cpu

# Build target
cpu: main_cpu

cpu_conf:
	mkdir -p $(SEQ_TARGET)

# FIXME possible to define a list of sources and then generate the commands below from the list instead of having explicit definitions? 
cpu_processing.o: cpu_conf $(SEQ_SRC)/processing.c
	$(CC) $(CFLAGS) -c $(SEQ_SRC)/processing.c -o $(SEQ_TARGET)/cpu_processing.o

cpu_device.o: cpu_conf $(SEQ_SRC)/device.c
	$(CC) $(CFLAGS) -c $(SEQ_SRC)/device.c -o $(SEQ_TARGET)/cpu_device.o

main_cpu: cpu_conf cpu_processing.o cpu_device.o $(COMMON_OBJECTS)
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(SRC_FOLDER)/main.cpp $(SEQ_TARGET)/cpu_processing.o $(SEQ_TARGET)/cpu_device.o $(COMMON_OBJECTS_PATH) -o $(SEQ_EXEC) 

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
	export OMP_NUM_THREADS 
# FIXME this could probably be deleted, test first  (OMP_NUM_THREADS)

openmp_processing.o: openmp_conf $(OPENMP_SRC)/processing.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOPENMP -c $(OPENMP_SRC)/processing.c  -o $(OPENMP_TARGET)/openmp_processing.o

openmp_device.o: cpu_conf $(OPENMP_SRC)/device.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOPENMP -c $(OPENMP_SRC)/device.c -o $(OPENMP_TARGET)/openmp_device.o

main_openmp: openmp_conf openmp_processing.o openmp_device.o $(COMMON_OBJECTS)
	mkdir -p $(BIN_PATH) 
	$(CC) $(CFLAGS) $(OMPFLAGS) -DOPENMP $(SRC_FOLDER)/main.cpp  $(OPENMP_TARGET)/openmp_processing.o $(OPENMP_TARGET)/openmp_device.o $(COMMON_OBJECTS_PATH) -o $(OPENMP_EXEC)

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

opencl_device.o: opencl_conf $(OPENCL_SRC)/device.c
	/bin/bash $(CONFIG_PATH)/CLHT.sh
	$(CC) -DOPENCL $(CFLAGS) -c $(OPENCL_SRC)/device.c  -o $(OPENCL_TARGET)/device.o $(OPFLAGS)

main_opencl: opencl_conf  opencl_device.o $(COMMON_OBJECTS)
	mkdir -p $(BIN_PATH) 
	$(CC) -DOPENCL $(CFLAGS) $(SRC_FOLDER)/main.cpp  $(OPENCL_TARGET)/device.o  $(COMMON_OBJECTS_PATH) -o $(OPENCL_EXEC)  $(OPFLAGS)

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

cuda_device.o: cuda_conf $(CUDA_SRC)/device.cu
	$(NVCC) $(NVCCFLAGS) $(CUFLAGS) -DCUDA -c $(CUDA_SRC)/device.cu -o $(CUDA_TARGET)/cuda_device.o

cuda_processing.o: cuda_conf $(CUDA_SRC)/processing.cu
	$(NVCC)  $(NVCCFLAGS) $(CUFLAGS) -DCUDA -c $(CUDA_SRC)/processing.cu  -o $(CUDA_TARGET)/cuda_processing.o

main_cuda: cuda_conf  cuda_device.o cuda_processing.o  $(COMMON_OBJECTS)
	mkdir -p $(BIN_PATH) 
	$(CC)  -DCUDA $(SRC_FOLDER)/main.cpp  $(CUDA_TARGET)/cuda_device.o $(CUDA_TARGET)/cuda_processing.o  $(COMMON_OBJECTS_PATH) -o $(CUDA_EXEC) $(CFLAGS) $(CUFLAGS)

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

hip_device.o: hip_conf $(HIP_SRC)/device.cpp
	$(HIPCC) $(HIPCCFLAGS) -DHIP -c $(HIP_SRC)/device.cpp -o $(HIP_TARGET)/hip_device.o $(CFLAGS)

hip_processing.o: hip_conf $(HIP_SRC)/processing.cpp
	$(HIPCC)  $(HIPCCFLAGS) -DHIP -c $(HIP_SRC)/processing.cpp -o $(HIP_TARGET)/lib_processing.o $(CFLAGS)

main_hip: hip_conf hip_device.o hip_processing.o  $(COMMON_OBJECTS)
	mkdir -p $(BIN_PATH)
	$(HIPCC) -DHIP $(SRC_FOLDER)/main.cpp -x none $(HIP_TARGET)/hip_device.o  $(HIP_TARGET)/lib_processing.o $(COMMON_OBJECTS_PATH) -o $(HIP_EXEC) $(CFLAGS) $(HIPCCFLAGS)

##################################### CLEAN ##################################
.PHONY: clean
clean:
	rm -f $(COM_TARGET)/*.o
	rm -f $(SEQ_TARGET)/*.o
	rm -f $(OPENMP_TARGET)/*.o
	rm -f $(OPENCL_TARGET)/*.o
	rm -f $(CUDA_TARGET)/*.o
	rm -f $(HIP_TARGET)/*.o
	rm -f $(EXEC_PATH)_*
