#!/bin/bash
BIN_PATH=../bin/2.1-data_compression/obpmark-2.1-data_compression

NUM_SAMPLES=1048576
NUM_BPS=16
REF_INTERVAL=4096
BLOCK_SIZE=64
INPUT_FILE=../data/input_data/2.1-compression/1024/2.1-compression-data_1024x1024.bin 

${BIN_PATH}_cpu -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_openmp -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_opencl -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_cuda -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t

NUM_SAMPLES=4194304
INPUT_FILE=../data/input_data/2.1-compression/2048/2.1-compression-data_2048x2048.bin 

${BIN_PATH}_cpu -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_openmp -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_opencl -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_cuda -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t

NUM_SAMPLES=16777216
INPUT_FILE=../data/input_data/2.1-compression/4096/2.1-compression-data_4096x4096.bin 

${BIN_PATH}_cpu -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_openmp -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_opencl -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t
${BIN_PATH}_cuda -s $NUM_SAMPLES -n $NUM_BPS -j $BLOCK_SIZE -r $REF_INTERVAL -f $INPUT_FILE -t

