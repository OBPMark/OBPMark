
#include "constants.h"

void print_usage(const char * appName);
int arguments_handler(int argc, char ** argv, ValidatorParameters* arguments_parameters);



int main(int argc, char *argv[]){
    ValidatorParameters arguments_parameters;
    // init arguments_parameters
    arguments_parameters.verification_non_stop = false;
    arguments_parameters.summary = false;
    arguments_parameters.range_verification = 1;
    arguments_parameters.number_of_values = 0;
    arguments_parameters.bit_depth = BIT_DEPTH;
    arguments_parameters.input_file_A[0] = '\0';
    arguments_parameters.input_file_B[0] = '\0';
    if (arguments_handler(argc, argv, &arguments_parameters) == ERROR_ARGUMENTS){
        return 1;
    }
    // init memory for both arrays with number_of_values and bit_depth
    unsigned long int *gold_refence = (unsigned long int *)malloc(arguments_parameters.number_of_values * sizeof(unsigned long int)); 
    unsigned long int *test_data = (unsigned long int *)malloc(arguments_parameters.number_of_values * sizeof(unsigned long int));
    // for read the binay file we need to know the bit size to read
    // open the files for reading
    FILE *file_A = fopen(arguments_parameters.input_file_A, "rb");
    FILE *file_B = fopen(arguments_parameters.input_file_B, "rb");
    // read the files
    if (arguments_parameters.bit_depth == 8){
        // read 8 bit
        for (unsigned long int i = 0; i < arguments_parameters.number_of_values; i++){
            fread(&gold_refence[i], sizeof(unsigned char), 1, file_A);
            fread(&test_data[i], sizeof(unsigned char), 1, file_B);
        }
    }else if (arguments_parameters.bit_depth == 16){
        // read 16 bit
        for (unsigned long int i = 0; i < arguments_parameters.number_of_values; i++){
            fread(&gold_refence[i], sizeof(unsigned short int), 1, file_A);
            fread(&test_data[i], sizeof(unsigned short int), 1, file_B);
        }
    }else if (arguments_parameters.bit_depth == 32){
        // read 32 bit
        for (unsigned long int i = 0; i < arguments_parameters.number_of_values; i++){
            fread(&gold_refence[i], sizeof(unsigned int), 1, file_A);
            fread(&test_data[i], sizeof(unsigned int), 1, file_B);
        }
    }else{
        // read 64 bit
        for (unsigned long int i = 0; i < arguments_parameters.number_of_values; i++){
            fread(&gold_refence[i], sizeof(unsigned long int), 1, file_A);
            fread(&test_data[i], sizeof(unsigned long int), 1, file_B);
        }
    }

    long unsigned int nerr = 0;
    long unsigned int max_diff = 0;
    long int diff = 0;
    // loop for compare the values
    for (unsigned long int i = 0; i < arguments_parameters.number_of_values; i++){
        // check if the values are different taking into account the range_verification that checks if the values are +/- range_verification
        if ((long int)(gold_refence[i]) < (long int)(test_data[i]) - (long int)(arguments_parameters.range_verification) ||  (long int)(gold_refence[i]) >  (long int)(test_data[i]) +  (long int)(arguments_parameters.range_verification)){
            if (arguments_parameters.verification_non_stop == false){
                printf("Error in the position %lu, the value is %lu and the expected value is %lu\n", i, test_data[i], gold_refence[i]);
                // break the loop
                break;
            }else{
                if(arguments_parameters.summary) 
                {
                    nerr++;
                    diff = (long int) gold_refence[i]- (long int) test_data[i];
                    if(diff < 0) diff = -diff;
                    if(diff > max_diff) max_diff = diff;
                }
                else printf("Error in the position %lu, the value is %lu and the expected value is %lu\n", i, test_data[i], gold_refence[i]);
            }
        }
    }

    if(arguments_parameters.summary)
        printf("Number of errors: %lu \nMaximum difference: %lu\n", nerr, max_diff);
    // free the memory
    free(gold_refence);
    free(test_data);
    

    return 0;

}



void print_usage(const char * appName)
{
	char str[12];
    printf("Usage: %s -i input_gold_reference input_to_validate -s size [-r value] [-n] [-b value]\n", appName);
    printf(" -i: input files \n");
    sprintf(str, "%d", RANGE_VERIFICATION_DEFAULT);
    printf(" -r: range of validation, default %s \n", str);
    sprintf(str, "%d", BIT_DEPTH);
    printf(" -b: bit depth, default %s \n", str);
    printf(" -s size: number of elements to read \n");
    printf(" -n: non stop verification, the program does not stop in the first discrepancy \n");
    printf(" -m: summary, if non stop verification, print the number of discrepancies and max difference \n");
	printf(" -h: print help information\n");

}



int arguments_handler(int argc, char ** argv, ValidatorParameters* arguments_parameters){

	for(unsigned int args = 1; args < argc; ++args)
	{
		switch (argv[args][1]) {
            case 'r' : args +=1; arguments_parameters->range_verification = atoi(argv[args]);break;
            case 'b' : args +=1; arguments_parameters->bit_depth = atoi(argv[args]);break;
            case 's' : args +=1; arguments_parameters->number_of_values = atoi(argv[args]);break;
            case 'n' : arguments_parameters->verification_non_stop = true;break;
            case 'm' : arguments_parameters->summary = true;break;
			case 'i' : args +=1;
					   strcpy(arguments_parameters->input_file_A,argv[args]);
					   args +=1;
					   strcpy(arguments_parameters->input_file_B,argv[args]);
					   break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

	}
	// check if the input files are not empty
    if (arguments_parameters->input_file_A[0] == '\0' || arguments_parameters->input_file_B[0] == '\0'){
        // print error message
        printf("Error: input files are not defined\n\n");
        // print usage
        print_usage(argv[0]);
        return ERROR_ARGUMENTS;
    }
    // check that bit depth is bigger than BIT_DEPTH
    if (arguments_parameters->bit_depth < BIT_DEPTH){
        // print error message
        printf("Error: bit depth is not valid\n\n");
        // print usage
        print_usage(argv[0]);
        return ERROR_ARGUMENTS;
    }
    // check that number of values is bigger than 0
    if (arguments_parameters->number_of_values == 0){
        // print error message
        printf("Error: number of values is not valid\n\n");
        // print usage
        print_usage(argv[0]);
        return ERROR_ARGUMENTS;
    }
	return OK_ARGUMENTS;
}
