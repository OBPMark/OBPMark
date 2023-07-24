#!/bin/bash
# This script compiles all of the subdirectories for the OBPMark project

# check the first argument to see if is a valid word from the list
# if no argument is given, then compile all

# list of valid words
valid_words="all cpu openmp cuda opencl hip allcuda allopencl allhip clean"

input_word=$1

# check if the input word is valid and witch one it is
for word in $valid_words
do
    if [ "$input_word" == "$word" ]; then
        if [ "all" == "$input_word" ]
        then
            echo "Compiling all"
            # compile all
            make clean 
            make cpu openmp cuda opencl hip
        elif [ "cpu" == "$input_word" ]
        then
            echo "Compiling cpu"
            # compile cpu
            make clean
            make cpu
        elif [ "openmp" == "$input_word" ]
        then
            echo "Compiling openmp"
            # compile openmp
            make clean
            make openmp
        elif [ "cuda" == "$input_word" ]
        then
            echo "Compiling cuda"
            # compile cuda
            make clean
            make cuda
        elif [ "opencl" == "$input_word" ]
        then
            echo "Compiling opencl"
            # compile opencl
            make clean
            make opencl
        elif [ "hip" == "$input_word" ]
        then
            echo "Compiling Hip"
            # compile hip
            make clean
            make hip
        elif [ "allcuda" == "$input_word" ]
        then
            echo "Compiling allcuda"
            # compile allcuda
            make clean
            make cpu cuda openmp
        elif [ "allopencl" == "$input_word" ]
        then
            echo "Compiling allopencl"
            # compile allopencl
            make clean
            make cpu opencl openmp
        
        elif [ "allhip" == "$input_word" ]
        then
            echo "Compiling allhip"
            # compile allhip
            make clean
            make cpu hip openmp
        elif [ "clean" == "$input_word" ]
        then
            echo "Cleaning"
            # clean
            make clean
        fi
        break
    fi
done
# check if the argument is empty
if [ -z "$input_word" ]; then
    echo "Compiling all"
    # compile all
    make clean 
    make cpu openmp cuda opencl hip
fi
