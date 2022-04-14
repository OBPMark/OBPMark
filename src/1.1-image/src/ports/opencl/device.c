/**
 * \file processing.c
 * \brief Benchmark #1.1 OpenCL implementation.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "benchmark.h"
#include "device.h"
#include "obpmark_time.h"

#include "GEN_processing.hcl"

// Define Frame status
#define E   0
#define R1  1
#define R2  2
#define FN  3
#define RO  4

#define NUMBER_STREAMS 3

///////////////////////////////////////////////////////////////////////////////////////////////
// OPENCL FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////
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
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        printf(" No platforms found. Check OpenCL installation!\n");
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platformB
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        printf(" No devices found. Check OpenCL installation!\n");
        exit(1);
    }
    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    image_data->context = new cl::Context(default_device);
    image_data->queue = new cl::CommandQueue(*image_data->context,default_device,NULL); //CL_QUEUE_PROFILING_ENABLE
    image_data->default_device = default_device;
    
    // events
    t->t_device_host = new cl::Event();
   

    // program
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_def_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    image_data->program = new cl::Program(*image_data->context,sources);
    if(image_data->program->build({image_data->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<image_data->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(image_data->default_device)<<"\n";
        exit(1);
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

    image_data->frames = new cl::Buffer(*image_data->context,CL_MEM_READ_WRITE , FRAMEBUFFERSIZE * (input_frames[0].h * input_frames[0].w * sizeof(uint16_t)));
    image_data->offsets = new cl::Buffer(*image_data->context,CL_MEM_READ_ONLY  ,offset_map->h * offset_map->w * sizeof(uint16_t));
    image_data->bad_pixels = new cl::Buffer(*image_data->context,CL_MEM_READ_ONLY  ,bad_pixel_map->h * bad_pixel_map->w * sizeof(uint8_t));
    image_data->gains = new cl::Buffer(*image_data->context,CL_MEM_READ_ONLY  , gain_map->h * gain_map->w * sizeof(uint16_t));
    image_data->image_output = new cl::Buffer(*image_data->context,CL_MEM_READ_WRITE , h_size* w_size * sizeof(uint32_t));

   
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

    const static uint8_t initial_frames = 5;
    // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
    // so uint16_t will be switch to unsigned short
    // and uint32_t  will be switch to unsigned int
    // and uint8_t will be switch to unsigned char
    // copy the initial frames to the device memory
    T_START(t->t_hots_device);
    for (int i = 0; i < initial_frames; ++i)
    {
        
        cl_buffer_region position = {(input_frames[i].h * input_frames[i].w * i) *sizeof(uint16_t_cl) ,input_frames[i].h * input_frames[i].w *sizeof(uint16_t_cl) };
        image_data->queue->enqueueWriteBuffer(image_data->frames->createSubBuffer(CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&position, NULL), CL_TRUE,0, input_frames[i].h * input_frames[i].w * sizeof(uint16_t_cl),input_frames[i].f,NULL,NULL);
    }
    // copy the offset map to the device memory
    image_data->queue->enqueueWriteBuffer(*image_data->offsets, CL_TRUE,0,correlation_table->h * correlation_table->w * sizeof(uint16_t_cl),correlation_table->f,NULL,NULL);
    // copy the gains map to the device memory
    image_data->queue->enqueueWriteBuffer(*image_data->gains, CL_TRUE,0,gain_correlation_map->h * gain_correlation_map->w * sizeof(uint16_t_cl),gain_correlation_map->f,NULL,NULL);
    // copy the bad pixels map to the device memory
    image_data->queue->enqueueWriteBuffer(*image_data->bad_pixels, CL_TRUE,0,bad_pixel_map->h * bad_pixel_map->w * sizeof(uint8_t_cl),bad_pixel_map->f,NULL,NULL);
    T_STOP(t->t_hots_device);
}


void process_benchmark(
	image_data_t *image_data,
	image_time_t *t,
    frame16_t *input_frames,
    unsigned int width,
    unsigned int height
	)
{
    uint8_t frame_status [FRAMEBUFFERSIZE];
    uint8_t number_frame_process = image_data->num_frames - 4; // -4 is because of the extra 4 images needed to be working
    uint8_t next_frame_to_copy = 5; //always copy starts at frame 5


    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;

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
    // create new queues
    /* Queue 0 focus in the copy of the input image, Queue 1 focus in the 1º part of the computation, Queue 2 focus in the 2º part*/
    cl::CommandQueue queues[NUMBER_STREAMS];
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
         queues[i] = cl::CommandQueue(*image_data->context,image_data->default_device,NULL);//CL_QUEUE_PROFILING_ENABLE
    }
    // status signals
    cl::Event evt_cp;
    cl::Event evt_p1;
    cl::Event evt_p2;

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

    T_START(t->t_test);
    // loop until we finish processing x amount of frames
    while (number_frame_process != 0){
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
                        // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
                        // so uint16_t will be switch to unsigned short
                        // and uint32_t  will be switch to unsigned int
                        cl_buffer_region position = {((width * height * i) *sizeof(uint16_t_cl)), ((width * height ) *sizeof(uint16_t_cl))};

                        queues[0].enqueueWriteBuffer(image_data->frames->createSubBuffer(CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&position, NULL), CL_TRUE,0, input_frames[next_frame_to_copy].h * input_frames[next_frame_to_copy].w * sizeof(unsigned short),input_frames[next_frame_to_copy].f,NULL,NULL);
                        queues[0].enqueueMarkerWithWaitList(NULL,&evt_cp);
                        cp = 1;
                        cp_pos = i;
                        ++next_frame_to_copy;
                        
                        break;
                    }
                }
            }
        }
        else
        {
            int* a = NULL; // I DO NOW WHY NEED TO BE SEPARATED
            // check if CP is empty
            if (evt_cp.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, a) == CL_COMPLETE){
                frame_status[cp_pos] = R1;
                cp = 0;
                evt_cp = cl::Event ();
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

                    cl::NDRange local_offset;
                    cl::NDRange global_offset;
                    if (width*height < BLOCK_SIZE * BLOCK_SIZE)
                    {
                        local_offset = cl::NullRange;
                        global_offset = cl::NDRange(width*height);
                    }
                    else
                    {
                        local_offset = cl::NDRange(x_local*y_local);
                        global_offset = cl::NDRange(width*height);
                    }
                    // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
                    // so uint16_t will be switch to unsigned short
                    // and uint32_t  will be switch to unsigned int
                    //cl_buffer_region position_P1 = {((width * height * i) *sizeof(unsigned short)), ((width * height) *sizeof(unsigned short))};
                    cl::Kernel kernel_offset=cl::Kernel(*image_data->program,"f_offset");
                    kernel_offset.setArg(0,*image_data->frames);
                    kernel_offset.setArg(1,i);
                    kernel_offset.setArg(2,*image_data->offsets);
                    kernel_offset.setArg(3, width * height);


                    queues[1].enqueueNDRangeKernel(kernel_offset,cl::NullRange,global_offset,local_offset, NULL, NULL);

                    cl::NDRange local_mask;
                    cl::NDRange global_mask;
                    if (width < BLOCK_SIZE || height < BLOCK_SIZE)
                    {
                        local_mask = cl::NullRange;
                        global_mask = cl::NDRange(width, height);
                    }
                    else
                    {
                        local_mask = cl::NDRange(x_local, y_local);
                        global_mask = cl::NDRange(width, height);
                    }
                    cl::Kernel kernel_mask=cl::Kernel(*image_data->program,"f_mask_replace");
                    kernel_mask.setArg(0,*image_data->frames);
                    kernel_mask.setArg(1,i);
                    kernel_mask.setArg(2,*image_data->bad_pixels);
                    kernel_mask.setArg(3, width);
                    kernel_mask.setArg(4, height);

                    queues[1].enqueueNDRangeKernel(kernel_mask,cl::NullRange,global_mask,local_mask, NULL, NULL);
                    queues[1].enqueueMarkerWithWaitList(NULL,&evt_p1);

                    r1p = 1;
                    r1p_pos = i;
                    break;

                }
                
            }
        }
        else 
        {
            // check if R1 is empty
            int* a = NULL; // I DO NOW WHY NEED TO BE SEPARATED
            // check if P1 is empty
            int result = evt_p1.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, a);
            int valid = CL_COMPLETE;
            if (evt_p1.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, a) == CL_COMPLETE){
                frame_status[r1p_pos] = R2;
                r1p = 0;
                evt_p1 = cl::Event ();
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
                        cl::NDRange local_scrub;
                        cl::NDRange global_scrub;
                        if (width < BLOCK_SIZE || height < BLOCK_SIZE)
                        {
                            local_scrub = cl::NullRange;
                            global_scrub = cl::NDRange(width, height);
                        }
                        else
                        {
                            local_scrub = cl::NDRange(x_local, y_local);
                            global_scrub = cl::NDRange(width, height);
                        }

                        
                        cl::Kernel kernel_scrub=cl::Kernel(*image_data->program,"f_scrub");
                        kernel_scrub.setArg(0,*image_data->frames);
                        kernel_scrub.setArg(1, i);
                        kernel_scrub.setArg(2, frame_i_0);
                        kernel_scrub.setArg(3, frame_i_1);
                        kernel_scrub.setArg(4, frame_i_2);
                        kernel_scrub.setArg(5, frame_i_3);
                        kernel_scrub.setArg(6, width);
                        kernel_scrub.setArg(7, height);

                        queues[2].enqueueNDRangeKernel(kernel_scrub,cl::NullRange,global_scrub,local_scrub, NULL, NULL);

                        // Second f_gain
                        cl::NDRange local_gain;
                        cl::NDRange global_gain;
                        if (width*height < BLOCK_SIZE * BLOCK_SIZE)
                        {
                            local_gain = cl::NullRange;
                            global_gain = cl::NDRange(width*height);
                        }
                        else
                        {
                            local_gain = cl::NDRange(x_local*y_local);
                            global_gain = cl::NDRange(width*height);
                        }
                        cl::Kernel kernel_gain=cl::Kernel(*image_data->program,"f_gain");
                        kernel_gain.setArg(0,*image_data->frames);
                        kernel_gain.setArg(1,i);
                        kernel_gain.setArg(2,*image_data->gains);
                        kernel_gain.setArg(3, width*height);
                        kernel_gain.setArg(4, width);
                        kernel_gain.setArg(5, height);

                        queues[2].enqueueNDRangeKernel(kernel_gain,cl::NullRange,global_gain,local_gain, NULL, NULL);
                        // Third f_2x2_bin_coadd

                        cl::NDRange local_coadd;
                        cl::NDRange global_coadd;
                        if (width < BLOCK_SIZE || height < BLOCK_SIZE)
                        {
                            local_coadd = cl::NullRange;
                            global_coadd = cl::NDRange(width/2, height/2);
                        }
                        else
                        {
                            local_coadd = cl::NDRange(x_local, y_local);
                            global_coadd = cl::NDRange(width/2, height/2);
                        }
                        cl::Kernel kernel_coadd=cl::Kernel(*image_data->program,"f_2x2_bin_coadd");
                        kernel_coadd.setArg(0,*image_data->frames);
                        kernel_coadd.setArg(1, i);
                        kernel_coadd.setArg(2,*image_data->image_output);
                        kernel_coadd.setArg(3, width/2);
                        kernel_coadd.setArg(4, height/2);
                        kernel_coadd.setArg(5, width/2);

                        queues[2].enqueueNDRangeKernel(kernel_coadd,cl::NullRange,global_coadd,local_coadd, NULL, NULL);

                        queues[2].enqueueMarkerWithWaitList(NULL,&evt_p2);
                        queues[2].finish();
                    }
                    
                }
            }
        }
        else
        {

            // check if R2 is empty
            int* a = NULL; // I DO NOW WHY NEED TO BE SEPARATED
            // check if P1 is empty
            if (evt_p2.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, a) == CL_COMPLETE){
                frame_status[r2p_pos] = FN;
                frame_status[r2p_pos_last_frame_pos] = RO;
                r2p = 0;
                --number_frame_process;
                evt_p2 = cl::Event ();
            }
            
        }
    }
    
   T_STOP(t->t_test);
}


void copy_memory_to_host(
	image_data_t *image_data,
	image_time_t *t,
	frame32_t *output_image
	)
{
    // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
    // so uint16_t will be switch to unsigned short
    // and uint32_t  will be switch to unsigned int
    image_data->queue->enqueueReadBuffer(*image_data->image_output,CL_TRUE,0,output_image->h * output_image->w * sizeof(uint32_t_cl),output_image->f, NULL, t->t_device_host);

}

void get_elapsed_time(
	image_data_t *image_data, 
	image_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{	

    t->t_device_host->wait();
	double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
    double host_to_device =   (t->t_hots_device) / ((double)(CLOCKS_PER_SEC / 1000)); 
    double device_to_host = t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_START>();

    if (csv_format)
	{
		
		printf("%.10f;%.10f;%.10f;\n", host_to_device, elapsed_time, device_to_host/ 1000000.0);
	}
	else if (database_format)
	{
		
		
		printf("%.10f;%.10f;%.10f;%ld;\n", host_to_device, elapsed_time, device_to_host/ 1000000.0, timestamp);
	}
	else if(verbose_print)
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (float) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", device_to_host/ 1000000.0);
	}
    /*device_object->memory_copy_host->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = device_object->memory_copy_device_a->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_device_a->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->memory_copy_device_b->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_device_b->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->memory_copy_device_c->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_device_c->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed =  (device_object->end.tv_sec - device_object->start.tv_sec) * 1000 + (device_object->end.tv_nsec - device_object->start.tv_nsec) / 1000000;
    elapsed_d_h = device_object->memory_copy_host->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_host->getProfilingInfo<CL_PROFILING_COMMAND_START>();


    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f miliseconds\n", elapsed );
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }*/
}



void clean(
	image_data_t *image_data, 
	image_time_t *t
	)
{

    delete image_data->context;
    delete image_data->queue;
    delete image_data->frames;
    delete image_data->offsets;
    delete image_data->gains;
    delete image_data->bad_pixels;
    delete image_data->binned_frame;
    delete image_data->image_output;
    delete image_data->scrub_mask;
    // pointer to memory
    delete t->t_device_host;
}
