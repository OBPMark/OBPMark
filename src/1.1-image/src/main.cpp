# include "benchmark_library.h"

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1


int arguments_handler(int argc, char ** argv,unsigned int *w_size, unsigned int *h_size, unsigned int *frames, unsigned int *bitsize, bool *csv_mode, bool *print_output);

int main(int argc, char **argv)
{
    // SRAND INIT
    srand (21121993);
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // ARGUMENTS PROCESSING  
    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int w_size = 0, h_size = 0, frames = 0, bitsize = 0;
    bool csv_mode = false, print_output = false;

    int resolution = arguments_handler(argc,argv,&w_size,&h_size,&frames,&bitsize,&csv_mode,&print_output);
    if ( resolution == ERROR_ARGUMENTS){exit(-1);}


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // CPU VARIBALES CREATION
    ///////////////////////////////////////////////////////////////////////////////////////////////
    unsigned int size_frame_list = w_size * h_size * frames;
    unsigned int size_frame = w_size * h_size;
    unsigned int mem_size_frame_list = size_frame_list * sizeof(int);
    unsigned int mem_size_frame = size_frame * sizeof(int);
    unsigned int mem_size_bad_map = size_frame * sizeof(bool);
    unsigned int size_reduction_image = (w_size/2) * (h_size/2);
    unsigned int mem_size_reduction_image = size_reduction_image* sizeof(int);
    int *input_frames = (int*) malloc(mem_size_frame_list);
    int *output_image = (int*) malloc(mem_size_reduction_image);
    // CALIBRATION AND CORRECTION DATA
    // Offset correction table
    int *correlation_table = (int*) malloc(mem_size_frame);
    // Bad pixel map
    bool *bad_pixel_map = (bool*) malloc(mem_size_frame);
    // Gain correction table
    int *gain_correlation_map = (int*) malloc(mem_size_frame);

    // selection of the 14 or 16 bits
    int  randnumber = 0;
    if (bitsize == MAXIMUNBITSIZE)
    {
        // UP TO 16 bits
        randnumber = 65535;
    }
    else if (bitsize == MINIMUNBITSIZE)
    {
        // UP TO 14 bits
        randnumber = 16383;
    }
    else
    {
        // DEFAULT 16 bits
        randnumber = 65535;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // CPU VARIBALES INIT
    ///////////////////////////////////////////////////////////////////////////////////////////////

    for (unsigned int frame_position = 0; frame_position < frames; ++frame_position )
    {
        for (unsigned int w_position = 0; w_position < w_size; ++w_position)
        {
            for (unsigned int h_position = 0; h_position < h_size; ++ h_position)
            {
                // Fill with random data
                input_frames[((h_position * h_size) + w_position) + (frame_position * h_size * w_size)] = (int)rand() % randnumber;

            }
        }
    }
    // offset correlation init 
    for (unsigned int w_position = 0; w_position < w_size; ++w_position)
    {
        for (unsigned int h_position = 0; h_position < h_size; ++ h_position)
        {
            // Fill with random data
            correlation_table[((h_position * h_size) + w_position)] = (int)rand() % randnumber;
        }
    }
    // gain correction table
    for (unsigned int w_position = 0; w_position < w_size; ++w_position)
    {
        for (unsigned int h_position = 0; h_position < h_size; ++ h_position)
        {
            // Fill with random data
            gain_correlation_map[((h_position * h_size) + w_position)] = (int)rand() % randnumber;
        }
    }
    // bad pixel correction
    for (unsigned int w_position = 0; w_position < w_size; ++w_position)
    {
        for (unsigned int h_position = 0; h_position < h_size; ++ h_position)
        {
            // Fill with random data
            bad_pixel_map[((h_position * h_size) + w_position)] = (rand() % MAXNUMBERBADPIXEL) < BADPIXELTHRESHOLD;
        }
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // GPU FUNCTION CALL
    ///////////////////////////////////////////////////////////////////////////////////////////////
    // INIT GPU
    // base object init
    DeviceObject *device_object = (DeviceObject *)malloc(sizeof(DeviceObject));
    char device[100] = "";
    // init the GPU
    init(device_object, 0,DEVICESELECTED, device);
    if (!csv_mode){
        printf("Using device: %s\n", device);
    }
    // init memory on the GPU
    device_memory_init(device_object, size_frame_list,size_reduction_image);
    copy_memory_to_device(device_object, correlation_table, gain_correlation_map, bad_pixel_map, size_frame);
    // process full frame list
    process_full_frame_list(device_object, input_frames,frames,size_frame,w_size,h_size);
    // copy back the image
    copy_memory_to_host(device_object,output_image,size_reduction_image);
    // get time
    get_elapsed_time(device_object,csv_mode);
    if(print_output)
    {
        // print output
        for (unsigned int h_position = 0; h_position < h_size/2; ++h_position)
        {
            
            for (unsigned int w_position = 0; w_position < w_size/2; ++w_position)
            {
                printf("%hu, ", output_image[(h_position * (h_size/2) + w_position)]);
            }
            printf("\n");
        }
    }

    // clean GPU
    clean(device_object);
    // clean CPU
    free(device_object);
    free(input_frames);
    free(output_image);
    free(correlation_table);
    free(gain_correlation_map);
    free(bad_pixel_map);

    return 0;
}

void print_usage(const char * appName)
{

    printf("Usage: %s -w [size] -h [size] -f [size] -b [size]\n", appName);
    printf(" -b size : bit size\n");
    printf(" -f size : number of frames\n");
    printf(" -w size : width of the input image in pixels\n");
	printf(" -h size : height of the input image in pixels \n");
    printf(" -c : print time in CSV\n");
    printf(" -o : print output\n");
}

int arguments_handler(int argc, char ** argv,unsigned int *w_size, unsigned int *h_size, unsigned int *frames, unsigned int *bitsize, bool *csv_mode, bool *print_output){
    if (argc < 4){
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    for(unsigned int args = 1; args < argc; ++args)
	{
        switch (argv[args][1]) {
            case 'w' : args +=1; *w_size = atoi(argv[args]);break;
            case 'h' : args +=1; *h_size = atoi(argv[args]);break;
            case 'f' : args +=1; *frames = atoi(argv[args]);break;
            case 'b' : args +=1; *bitsize = atoi(argv[args]);break;
            case 'c' : *csv_mode = true;break;
            case 'o' : *print_output = true;break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

    }
    if ( *w_size < MINIMUNWSIZE){
		printf("-w need to be set and bigger than or equal to %d\n\n", MINIMUNWSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    if ( *h_size < MINIMUNHSIZE){
		printf("-h need to be set and bigger than or equal to %d\n\n", MINIMUNHSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    if ( *frames < MINIMUNFRAMES){
		printf("-f need to be set and bigger than or equal to %d\n\n", MINIMUNFRAMES);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    if ( *bitsize != MINIMUNBITSIZE && *bitsize != MAXIMUNBITSIZE){
		printf("-b need to be set and be %d or %d\n\n", MINIMUNBITSIZE, MAXIMUNBITSIZE);
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}

	return OK_ARGUMENTS;
}