#!/bin/bash
BIN_PATH=../bin/3.1-aes_encryption/obpmark-3.1-aes_encryption

DATA_LENGTH=1048576
KEY_LENGTH=128

# FIXME file reading does not work, using random data temporarily

${BIN_PATH}_cpu -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_openmp -l $DATA_LENGTH -k $KEY_LENGTH -r -t
#${BIN_PATH}_opencl -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_cuda -l $DATA_LENGTH -k $KEY_LENGTH -r -t

DATA_LENGTH=4194304
KEY_LENGTH=192

${BIN_PATH}_cpu -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_openmp -l $DATA_LENGTH -k $KEY_LENGTH -r -t
#${BIN_PATH}_opencl -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_cuda -l $DATA_LENGTH -k $KEY_LENGTH -r -t

DATA_LENGTH=16777216
KEY_LENGTH=256

${BIN_PATH}_cpu -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_openmp -l $DATA_LENGTH -k $KEY_LENGTH -r -t
#${BIN_PATH}_opencl -l $DATA_LENGTH -k $KEY_LENGTH -r -t
${BIN_PATH}_cuda -l $DATA_LENGTH -k $KEY_LENGTH -r -t

