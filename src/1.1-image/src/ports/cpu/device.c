/**
 * \file device.c
 * \brief Benchmark #1.1 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez (BSC)
 */
#include "device.h"
#include "processing.h"

void init(DeviceObject *device_object, char* device_name)
{
    init(device_object, 0,0, device_name);
}


void init(DeviceObject *device_object, int platform ,int device, char* device_name)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(DeviceObject *device_object, unsigned int size_image, unsigned int size_reduction_image)
{
    device_object->image_output = (int*) malloc (size_reduction_image * sizeof(int));
    device_object->processing_image = (int*) malloc (size_image * sizeof(int));
    device_object->processing_image_error_free = (int*) malloc (size_image * sizeof(int));
    return true;
}


void copy_memory_to_device(DeviceObject *device_object, int* correlation_table, int* gain_correlation_map, bool* bad_pixel_map , unsigned int size_image)
{
    device_object->correlation_table = correlation_table;
    device_object->gain_correlation_map = gain_correlation_map;
    device_object->bad_pixel_map = bad_pixel_map;
}


void copy_frame_to_device(DeviceObject *device_object, int* input_data, unsigned int size_image, unsigned int frame)
{
    
    //for (unsigned int position = 0; position < 1; ++position)
    //{
        //device_object->image_input[0] = input_data[0 + (frame * size_image)] ;
    //}
    device_object->image_input = input_data + (frame * size_image) ;
    
}

void process_full_frame_list (DeviceObject *device_object,int* input_frames,unsigned int frames, unsigned int size_frame,unsigned int w_size, unsigned int h_size){
    // Start timer
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start);
   
   for (unsigned int frame = 0; frame < frames; ++frame )
    {
        // copy image
        copy_frame_to_device(device_object, input_frames, size_frame, frame);
        // process image
        process_image(device_object, w_size, h_size, frame);
    }

    // End timer
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end);
}


void process_image(DeviceObject *device_object, unsigned int w_size, unsigned int h_size, unsigned int frame)
{
	benchmark_params_t p; // FIXME merge parameter structs
	p.offsets = device_object->correlation_table;  // FIXME offset map is incorrectly named
	p.gains = device_object->gain_correlation_table; // FIXME wrongly named... not a correlation table.
	// FIXME should have one struct with all the data... and nothing else... other things like timers in other struct 

	// FIXME ESA code uses 2D arrays, BSC code uses 1D arrays -- any difference? ... maybe go with 1D arrays, since that is what is used in OpenCL, CUDA, etc. 
	frame16_t frame = device_object->image_input;
	frame	

	proc_image_frame(benchmark_params_t *p, frame16_t *frame, unsigned int frame_i)

	// FIXME indent
	// FIXME cleanup below


    const unsigned int size_image = w_size * h_size;

    // Image offset correlation gain correction
    for(unsigned int i = 0; i < size_image; ++i)
    {
        //device_object->processing_image[i] = (device_object->image_input[i]);
        device_object->processing_image[i] = (device_object->image_input[i] - device_object->correlation_table[i]) * device_object->gain_correlation_map[i];
    }


    // Bad pixel correlation
    for(unsigned int y = 0; y < h_size; ++y)
    {
        for(unsigned int x = 0; x < w_size; ++x)
        {
            
            // Lots and lots of bifurcations: Suboptimal
            if (device_object->bad_pixel_map[y * h_size + x])
            {
                if (x == 0 && y == 0)
                {
                    // TOP left
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[y * h_size +  (x +1)] + device_object->processing_image[(y +1) * h_size +  (x +1) ] + device_object->processing_image[(y +1) * h_size + x  ])/3;
                }
                else if (x == 0 && y == h_size)
                {
                    // Top right
                    device_object->processing_image_error_free[y * h_size + x] = (device_object->processing_image[y * h_size +  (x -1)] + device_object->processing_image[(y -1) * h_size +  (x -1)] + device_object->processing_image[(y -1) * h_size + x ])/3;
                }
                else if(x == w_size && y == 0)
                {
                    //Bottom left
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[(y -1) * h_size +  x] + device_object->processing_image[(y -1) * h_size +  (x + 1)] + device_object->processing_image[y * h_size +  (x +1)])/3;
                }
                else if (x == w_size && y == h_size)
                {
                    // Bottom right
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[(y -1) * h_size +  (x -1)] + device_object->processing_image[(y -1) * h_size +  x ] + device_object->processing_image[y * h_size +  (x -1)])/3;
                }
                else if (y == 0)
                {
                    // Top Edge
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[y * h_size +  (x -1) ] + device_object->processing_image[y * h_size +  (x +1) ] + device_object->processing_image[(y +1) * h_size +  x ])/3;
                }
                else if (x == 0)
                {
                    //  Left Edge
                    device_object->processing_image_error_free[y * h_size + x] = (device_object->processing_image[(y -1) * h_size +  x ] + device_object->processing_image[y * h_size +  (x +1) ] + device_object->processing_image[(y +1) * h_size +  x ])/3;
                }
                else if (x == w_size)
                {
                    //  Right Edge
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[(y -1) * h_size +  x ] + device_object->processing_image[y * h_size +  (x -1) ] + device_object->processing_image[(y +1) * h_size +  x ])/3;
                }
                else if (y == h_size)
                {
                    // Bottom Edge
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[(y -1) * h_size +  x ] + device_object->processing_image[y * h_size +  (x -1) ] + device_object->processing_image[y * h_size +  (x +1)])/3;
                }
                else
                {
                    // Standart Case
                    device_object->processing_image_error_free[y * h_size + x ] = (device_object->processing_image[y * h_size +  (x -1)] + device_object->processing_image[y * h_size +  (x -1) ] + device_object->processing_image[(y +1) * h_size +  x  ] +  device_object->processing_image[(y +1) * h_size +  x  ])/4;
                }
            }
            else
            {
                device_object->processing_image_error_free[y * h_size + x ] = device_object->processing_image[y * h_size +  x];
            }
            //printf("%hu, ", device_object->processing_image_error_free[(y * (h_size) + x)]);
        }
        //printf("\n");
        //printf("\n");
    }
    //printf("\n");

    // Spatial Binning Temporal Binning
    const unsigned int w_size_half = w_size/2;
    const unsigned int h_size_half = h_size/2;
    // #pragma omp parallel for
    for(unsigned int y = 0; y < h_size_half; ++y)
    {
        for(unsigned int x = 0; x < w_size_half; ++x)
        {
            device_object->image_output[y * h_size_half + x ] += device_object->processing_image_error_free[ (2*y)* (h_size_half*2) + (2 *x) ] + device_object->processing_image_error_free[(2*y)* (h_size_half*2) + (2 *(x+1))  ] + device_object->processing_image_error_free[(2*(y+1))* (h_size_half*2) + (2 *x) ] + device_object->processing_image_error_free[(2*(y+1))* (h_size_half*2) + (2 *(x+1)) ];
        }
    }
}


void copy_memory_to_host(DeviceObject *device_object, int* output_image, unsigned int size_image)
{
    memcpy(output_image, &device_object->image_output[0], sizeof(int) * size_image);
}


float get_elapsed_time(DeviceObject *device_object, bool csv_format)
{
    float elapsed =  (device_object->end.tv_sec - device_object->start.tv_sec) * 1000 + (device_object->end.tv_nsec - device_object->start.tv_nsec) / 1000000;
    if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (float) 0, elapsed, (float) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f miliseconds\n", (float) 0);
		printf("Elapsed time kernel: %.10f miliseconds\n", elapsed);
		printf("Elapsed time Device->Host: %.10f miliseconds\n", (float) 0);
    }
	return elapsed;
}


void clean(DeviceObject *device_object)
{
	free(device_object->image_output);
    free(device_object->processing_image_error_free);
    free(device_object->processing_image);
}
