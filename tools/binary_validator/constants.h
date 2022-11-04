#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1

#define RANGE_VERIFICATION_DEFAULT 1
#define BIT_DEPTH 8

struct ValidatorParameters{
    bool verification_non_stop = false;
    bool summary = false;
    unsigned int  range_verification = 0;
    unsigned int  bit_depth = 0;
    unsigned long int number_of_values = 0;
	char input_file_A[100] = "";
	char input_file_B[100] = "";
};



#endif  // CONSTANTS_H
