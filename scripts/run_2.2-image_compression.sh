#!/bin/bash
BIN_PATH=../bin/2.2-image_compression/obpmark-2.2-image_compression

IMAGE_WIDTH=1024
IMAGE_HEIGHT=1024
NUM_BPS=16
BLOCK_SIZE=64
INPUT_FILE=../data/input_data/2.1-compression/1024/2.1-compression-data_1024x1024.bin 

# FIXME uses wrong input files for now
# FIXME should also include all profiles as defined in the OBPMark TN (running with the float option)

${BIN_PATH}_cpu -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_openmp -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_opencl -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_cuda -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t

IMAGE_WIDTH=2048
IMAGE_HEIGHT=2048
INPUT_FILE=../data/input_data/2.1-compression/2048/2.1-compression-data_2048x2048.bin 

${BIN_PATH}_cpu -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_openmp -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_opencl -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_cuda -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t

IMAGE_WIDTH=4096
IMAGE_HEIGHT=4096
INPUT_FILE=../data/input_data/2.1-compression/4096/2.1-compression-data_4096x4096.bin 

${BIN_PATH}_cpu -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_openmp -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_opencl -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t
${BIN_PATH}_cuda -w $IMAGE_WIDTH -h $IMAGE_HEIGHT -b $NUM_BPS -s $BLOCK_SIZE -f $INPUT_FILE -t

