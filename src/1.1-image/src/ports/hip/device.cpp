#include "hip/hip_runtime.h"
/**
 * \file device.cu
 * \brief Benchmark #1.1 CUDA version device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */

#include "device.h"
#include "processing.h"

// Define Frame status
#define E   0
#define R1  1
#define R2  2
#define FN  3
#define RO  4


#define NUMBER_STREAMS 3
/* stream 0 focus in the copy of the input image, stream 1 focus in the 1º part of the computation, stream 2 focus in the 2º part*/
hipStream_t cuda_streams[NUMBER_STREAMS]; 

void init(
	image_data_t *image_data,
	image_time_t *t,
	char *device_name
	)
{
    init(image_data,t, 0,0, device_name);
}



void init(
	image_data_t *image_data,
	image_time_t *t,
	int platform,
	int device,
	char *device_name
	)
{
    hipSetDevice(device);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    strcpy(device_name,prop.name);
    // TODO verfy that these are the required events for timming
    //event create 
    t->start_test = new hipEvent_t;
    t->stop_test = new hipEvent_t;
    t->start_memory_copy_device = new hipEvent_t;
    t->stop_memory_copy_device = new hipEvent_t;
    t->start_memory_copy_host = new hipEvent_t;
    t->stop_memory_copy_host= new hipEvent_t;

    // init streams
    for(unsigned int x = 0; x < NUMBER_STREAMS; ++x){
        // this makes the priority to have a image copy and the 1º part of the computation is done first
        hipStreamCreateWithPriority(&cuda_streams[x],hipStreamDefault,x+1);
    }


}
bool device_memory_init(
	image_data_t* image_data,
	frame16_t* input_frames,
	frame16_t* offset_map, 
	frame8_t* bad_pixel_map, 
	frame16_t* gain_map,
	unsigned int w_size,
	unsigned int h_size
	)
{	
    // allocate memory for the frame buffer
    hipError_t err = hipSuccess;
    err = hipMalloc((void **)&image_data->frames, FRAMEBUFFERSIZE * (input_frames[0].h * input_frames[0].w * sizeof(uint16_t)));
 
    if (err != hipSuccess)
    {
        return false;
    }

    // allocate memory for the offset map
    err = hipMalloc((void **)&image_data->offsets, offset_map->h * offset_map->w * sizeof(uint16_t) );
    if (err != hipSuccess)
    {
        return false;
    }
    // allocate memory for the gains map
    err = hipMalloc((void **)&image_data->gains, gain_map->h * gain_map->w * sizeof(uint16_t) );
    if (err != hipSuccess)
    {
        return false;
    }
    // allocate memory for the bad pixels map
    err = hipMalloc((void **)&image_data->bad_pixels, bad_pixel_map->h * bad_pixel_map->w * sizeof(uint8_t));
    if (err != hipSuccess)
    {
        return false;
    }
    // allocate memory for the image output
    err = hipMalloc((void **)&image_data->image_output, h_size* w_size * sizeof(uint32_t));
    if (err != hipSuccess)
    {
        return false;
    }

    return true;
}



void copy_memory_to_device(
	image_data_t *image_data,
	image_time_t *t,
	frame16_t *input_frames,
	frame16_t *correlation_table,
	frame16_t *gain_correlation_map,
	frame8_t *bad_pixel_map
	)
{
    hipEventRecord(*t->start_memory_copy_device);
    // copy the initial frames to the device memory
    const static uint8_t initial_frames = 5;
    for (int i = 0; i < initial_frames; ++i)
    {
        hipMemcpy(image_data->frames + ((input_frames[i].h * input_frames[i].w) * i) , input_frames[i].f,  input_frames[i].h * input_frames[i].w * sizeof(uint16_t) , hipMemcpyHostToDevice);
    }
    // copy the offset map to the device memory
    hipMemcpy(image_data->offsets , correlation_table->f,  correlation_table->h * correlation_table->w * sizeof(uint16_t) , hipMemcpyHostToDevice);
    // copy the gains map to the device memory
    hipMemcpy(image_data->gains, gain_correlation_map->f, gain_correlation_map->h * gain_correlation_map->w * sizeof(uint16_t), hipMemcpyHostToDevice);
    // copy the bad pixels map to the device memory
    hipMemcpy(image_data->bad_pixels, bad_pixel_map->f, bad_pixel_map->h * bad_pixel_map->w * sizeof(uint8_t), hipMemcpyHostToDevice);



    hipEventRecord(*t->stop_memory_copy_device);

}


void process_benchmark(
	image_data_t *image_data,
	image_time_t *t,
    frame16_t *input_frames,
    unsigned int width,
    unsigned int height
	)
{
    /*
        In order to mitigate memory usage in the device, we use a frame buffer, the size of it is defined by FRAMEBUFFERSIZE
        This buffer work in a circular way.
    */

    uint8_t frame_status [FRAMEBUFFERSIZE];
    uint8_t number_frame_process = image_data->num_frames - 4; // -4 is because of the extra 4 images needed to be working
    uint8_t next_frame_to_copy = 5; //always copy starts at frame 5


    // status of the pipes TODO this could be change to support more than one pipe per type
    uint8_t cp = 0;
    uint8_t cp_pos = 0;
    uint8_t r1p = 0;
    uint8_t r1p_pos = 0;
    uint8_t r2p = 0;
    uint8_t r2p_pos = 0;
    uint8_t r2p_pos_last_frame_pos = 0;
    // frame positions
    uint8_t frame_i_ready = 0;
    uint8_t frame_i_0 = 0;
    uint8_t frame_i_1 = 0;
    uint8_t frame_i_2 = 0;
    uint8_t frame_i_3 = 0;
    // signals
    hipError_t copy_stream;
    hipError_t pipe1_stream;
    hipError_t pipe2_stream;
    /*
      Status legend
      int value | Code name | Description
        0       | E         | Empty position ,free to be fill up
        1       | R1        | Frame ready for first part of processing
        2       | R2        | Frame ready for second part of processing
        3       | FN        | Frame finish processing but needed for other frame
        4       | RO        | Frame ready for be overwritten 
    */
    // init frame_status, this inits the frame status to the initial status
    for (unsigned int i = 0; i < FRAMEBUFFERSIZE; ++i)
    {
        frame_status[i] = 0;
    }
    // the first 5 frames will be always be FN, FN, R2, R2, R1   
    frame_status[0] = FN;
    frame_status[1] = FN;
    frame_status[2] = R2;
    frame_status[3] = R2;
    frame_status[4] = R1;
    hipEventRecord(*t->start_test);
    // loop until we finish processing x amount of frames
    while (number_frame_process != 0){

        //printf("%d-%d-%d-%d-%d-%d:  %d,%d,%d: %d\n", frame_status[0],frame_status[1],frame_status[2],frame_status[3],frame_status[4],frame_status[5], cp,r1p,r2p, number_frame_process);
        // fist check for empty spaces for copy data
        // fist check if is any left
        /*
            CHECK FOR COPY IMAGE 
        */
        if (cp == 0)
        {
            if (next_frame_to_copy < image_data->num_frames)
            {
                // check for empty spaces for copy data
                for (unsigned int i = 0; i < FRAMEBUFFERSIZE; ++i)
                {
                    if (frame_status[i] == E || frame_status[i] == RO){
                        // call function to copy
                        hipMemcpyAsync(image_data->frames + ((height * width) * i), input_frames[next_frame_to_copy].f, input_frames[next_frame_to_copy].h * input_frames[next_frame_to_copy].w * sizeof(uint16_t), hipMemcpyHostToDevice,cuda_streams[0]);
                        cp = 1;
                        cp_pos = i;
                        // call the update func
                        // cudaLaunchHostFunc(cuda_streams[0], &update_frame_status_async, &frame_status,i,R1); 
                        ++next_frame_to_copy;
                        
                        break;
                    }
                }
            }
        }
        else
        {
            // check if CP is empty
            copy_stream = hipStreamQuery(cuda_streams[0]);
            if(copy_stream == hipSuccess){
                frame_status[cp_pos] = R1;
                cp = 0;
            }
            else if (pipe1_stream == hipErrorIllegalAddress)
            {
                printf("Error Illegal Access memory in the GPU Copy\n");
                exit(-1);
            }
        }
        

        /*
            CHECK FOR FRAME GOING TO THE FIRST PIPELINE
        */

        // second check for R1 frames to be launch to the pipe
        if (r1p == 0)
        {
            // R1 pipe empty
            for (unsigned int i = 0; i < FRAMEBUFFERSIZE; ++i)
            {
                if ( frame_status[i] == R1)
                {
                    dim3 dimBlock_offset(BLOCK_SIZE*BLOCK_SIZE);
                    dim3  dimGrid_offset(ceil(float(width*height)/dimBlock_offset.x));

                    hipLaunchKernelGGL(f_offset, dim3(dimGrid_offset), dim3(dimBlock_offset), 0, cuda_streams[1] , image_data->frames + ((height * width) * i), image_data->offsets, width * height);

                    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
                    dim3 dimGrid(ceil(float(height)/dimBlock.x), ceil(float(width)/dimBlock.y));

                    hipLaunchKernelGGL(f_mask_replace, dim3(dimGrid), dim3(dimBlock), 0, cuda_streams[1] , image_data->frames + ((height * width ) * i), image_data->bad_pixels, width, height);
                    r1p = 1;
                    r1p_pos = i;
                    break;

                }
                
            }
        }
        else 
        {
            // check if R1 is empty
            pipe1_stream = hipStreamQuery(cuda_streams[1]);
            if(pipe1_stream == hipSuccess){
                frame_status[r1p_pos] = R2;
                r1p = 0;
            }
            else if (pipe1_stream == hipErrorIllegalAddress)
            {
                printf("Error Illegal Access memory in the GPU Pipe 1\n");
                exit(-1);
            }
        }
        

        /*
            CHECK FOR FRAME GOING TO THE SECOND PIPELINE
        */
        
        if (r2p == 0)
        {
            // third check for ready R2 to be launch to the pipe
            for (unsigned int i = 0; i < FRAMEBUFFERSIZE; ++i)
            {
                if ( frame_status[i] == R2)
                {
                    // we have one frame ready for going to R2, but the frames before are ready?
                    if (i >= 2 && i + 2 < FRAMEBUFFERSIZE)
                    {
                        if(frame_status[i-2] == FN && frame_status[i-1] == FN && frame_status[i+1] == R2 && frame_status[i+2] == R2)
                        {
                            frame_i_ready = 1;
                            frame_i_0 = i - 2;
                            frame_i_1 = i - 1;
                            frame_i_2 = i + 1;
                            frame_i_3 = i + 2;
                        }
                        else
                        {
                            frame_i_ready = 0;
                        }
                    }
                    else if ( i == 1 )
                    {
                        if(frame_status[FRAMEBUFFERSIZE - 1] == FN && frame_status[i-1] == FN && frame_status[i+1] == R2 && frame_status[i+2] == R2)
                        {
                            frame_i_ready = 1;
                            frame_i_0 = FRAMEBUFFERSIZE - 1;
                            frame_i_1 = i - 1;
                            frame_i_2 = i + 1;
                            frame_i_3 = i + 2;
                        }
                        else
                        {
                            frame_i_ready = 0;
                        }


                    }
                    else if ( i == 0)
                    {
                        if(frame_status[FRAMEBUFFERSIZE - 2] == FN && frame_status[FRAMEBUFFERSIZE - 1] == FN && frame_status[i+1] == R2 && frame_status[i+2] == R2)
                        {
                            frame_i_ready = 1;
                            frame_i_0 = FRAMEBUFFERSIZE - 2;
                            frame_i_1 = FRAMEBUFFERSIZE - 1;
                            frame_i_2 = i + 1;
                            frame_i_3 = i + 2;
                        }
                        else
                        {
                            frame_i_ready = 0;
                        }

                    }
                    else if (i + 1 == FRAMEBUFFERSIZE)
                    {
                        if(frame_status[i-2] == FN && frame_status[i-1] == FN && frame_status[0] == R2 && frame_status[1] == R2)
                        {
                            frame_i_ready = 1;
                            frame_i_0 = i - 2;
                            frame_i_1 = i - 1;
                            frame_i_2 = 0;
                            frame_i_3 = 1;
                        }
                        else
                        {
                            frame_i_ready = 0;
                        }
                    }
                    else if (i + 2 == FRAMEBUFFERSIZE)
                    {
                        if(frame_status[i-2] == FN && frame_status[i-1] == FN && frame_status[i+1] == R2 && frame_status[0] == R2)
                        {
                            frame_i_ready = 1;
                            frame_i_0 = i - 2;
                            frame_i_1 = i - 1;
                            frame_i_2 = i + 1;
                            frame_i_3 = 0;
                        }
                        else
                        {
                            frame_i_ready = 0;
                        }
                    }
                    else
                    {
                        frame_i_ready = 0; 
                    }

                    // now check if everting is ready
                    if (frame_i_ready == 1)
                    {
                        // i go to R2
                        r2p = 1;
                        r2p_pos = i;
                        r2p_pos_last_frame_pos = frame_i_0;
                        // star launching the execution
                        // First f_scrub
                        dim3 dimBlock_scrub(BLOCK_SIZE, BLOCK_SIZE);
                        dim3 dimGrid_scrub(ceil(float(width)/dimBlock_scrub.x), ceil(float(height)/dimBlock_scrub.y));

                        hipLaunchKernelGGL(f_scrub, dim3(dimGrid_scrub), dim3(dimBlock_scrub), 0, cuda_streams[2], image_data->frames +  ((height * width ) * i), image_data->frames +  ((height * width ) * frame_i_0),image_data->frames +  ((height * width ) * frame_i_1),image_data->frames +  ((height * width ) * frame_i_2),image_data->frames +  ((height * width ) * frame_i_3),width,height);
                        // Second f_gain
                        dim3 dimBlock_gain(BLOCK_SIZE*BLOCK_SIZE);
                        dim3  dimGrid_gain(ceil(float(width*height)/dimBlock_gain.x));

                        hipLaunchKernelGGL(f_gain, dim3(dimGrid_gain), dim3(dimBlock_gain), 0, cuda_streams[2], image_data->frames +  ((height * width ) * i), image_data->gains, width*height);
                        // Third f_2x2_bin_coadd
                        dim3 dimBlock_bin_coadd(BLOCK_SIZE, BLOCK_SIZE);
                        dim3 dimGrid_bin_coadd(ceil(float(width/2)/dimBlock_bin_coadd.x), ceil(float(height/2)/dimBlock_bin_coadd.y));

                        hipLaunchKernelGGL(f_2x2_bin_coadd, dim3(dimGrid_bin_coadd), dim3(dimBlock_bin_coadd), 0, cuda_streams[2], image_data->frames +  ((height * width ) * i),image_data->image_output, width/2, height/2, width/2);

                    }
                    
                }
            }
        }
        else
        {
            // check if R1 is empty
            pipe2_stream = hipStreamQuery(cuda_streams[2]);
            if(pipe2_stream == hipSuccess){
                frame_status[r2p_pos] = FN;
                frame_status[r2p_pos_last_frame_pos] = RO;
                r2p = 0;
                --number_frame_process;
            }
            else if (pipe1_stream == hipErrorIllegalAddress)
            {
                printf("Error Illegal Access memory in the GPU  Pipe 2\n");
                exit(-1);
            }
        }
    }
    hipEventRecord(*t->stop_test);

}

void copy_memory_to_host(
	image_data_t *image_data,
	image_time_t *t,
	frame32_t *output_image
	)
{
    hipEventRecord(*t->start_memory_copy_host);
    hipMemcpy(output_image->f, image_data->image_output, output_image->h * output_image->w * sizeof(uint32_t), hipMemcpyDeviceToHost);
    hipEventRecord(*t->stop_memory_copy_host);

}

void get_elapsed_time(
	image_data_t *image_data, 
	image_time_t *t, 
	print_info_data_t *benchmark_info,
	long int timestamp
	)
{	

    hipEventSynchronize(*t->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    hipEventElapsedTime(&milliseconds_h_d, *t->start_memory_copy_device, *t->stop_memory_copy_device);
    // kernel time
    hipEventElapsedTime(&milliseconds, *t->start_test, *t->stop_test);
    //  memory transfer time device-host
    hipEventElapsedTime(&milliseconds_d_h, *t->start_memory_copy_host, *t->stop_memory_copy_host);
    

    print_execution_info(benchmark_info, true, timestamp,milliseconds_h_d,milliseconds,milliseconds_d_h);

}
/*
void get_elapsed_time(DeviceObject *device_object, bool csv_format){
    hipEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    hipEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    hipEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    hipEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f ms\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f ms\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f ms\n", milliseconds_d_h);
    }
}
*/
void clean(
	image_data_t *image_data, 
	image_time_t *t
	)
{
    // delete cuda memory
    hipError_t err = hipSuccess;

    err = hipFree(image_data->offsets);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector offsets (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    
    err = hipFree(image_data->gains);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector gains (error code %s)!\n", hipGetErrorString(err));
        return;
    }
    err = hipFree(image_data->bad_pixels);

    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector bad_pixels (error code %s)!\n", hipGetErrorString(err));
        return;
    }


    err = hipFree(image_data->frames);
    if (err != hipSuccess)
    {
        fprintf(stderr, "Failed to free device vector frames (error code %s)!\n", hipGetErrorString(err));
        return;
    }
   
    
    // delete events
    delete t->start_test;
    delete t->stop_test;
    delete t->start_memory_copy_device;
    delete t->stop_memory_copy_device;
    delete t->start_memory_copy_host;
    delete t->stop_memory_copy_host;

    // clean streams
    for(unsigned int x = 0; x < NUMBER_STREAMS ; ++x){
        hipStreamDestroy(cuda_streams[x]);
    }



}