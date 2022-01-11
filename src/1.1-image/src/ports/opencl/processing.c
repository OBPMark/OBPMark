/**
 * \file processing.c
 * \brief Benchmark #1.1 OpenCL implementation.
 * \author Ivan Rodriguez (BSC)
 */
#include "benchmark.h"
#include "benchmark_opencl.h"
#include "device.h"

#include "GEN_kernel.hcl"

///////////////////////////////////////////////////////////////////////////////////////////////
// OPENCL FUNCTIONS
///////////////////////////////////////////////////////////////////////////////////////////////
void init(DeviceObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}


void init(DeviceObject *device_object, int platform ,int device, char* device_name){
	//get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platformB
    std::vector<cl::Device> all_devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if(all_devices.size()==0){
        std::cout<<" No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Device default_device=all_devices[device];
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device.getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    device_object->context = new cl::Context(default_device);
    device_object->queue = new cl::CommandQueue(*device_object->context,default_device,CL_QUEUE_PROFILING_ENABLE);
    device_object->default_device = default_device;
    
    // events
    device_object->memory_copy_device_a = new cl::Event; 
    device_object->memory_copy_device_b = new cl::Event;
    device_object->memory_copy_device_c = new cl::Event;
    device_object->memory_copy_host = new cl::Event;

    // program
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    device_object->program = new cl::Program(*device_object->context,sources);
    if(device_object->program->build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<device_object->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
    
}

bool device_memory_init(DeviceObject *device_object, unsigned int size_image, unsigned int size_reduction_image){

    device_object->image_input = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE , size_image * sizeof(int));
    device_object->processing_image = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE , size_image * sizeof(int));
    device_object->processing_image_error_free = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE , size_image * sizeof(int));
    device_object->image_output = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE , size_reduction_image * sizeof(int));
    device_object->correlation_table = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY , size_image * sizeof(int));
    device_object->gain_correlation_map = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY , size_image * sizeof(int));
    device_object->bad_pixel_map = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY , size_image * sizeof(bool));
    return true;
}
void copy_memory_to_device(DeviceObject *device_object, int* correlation_table, int* gain_correlation_map, bool* bad_pixel_map , unsigned int size_image){
    

    device_object->queue->enqueueWriteBuffer(*device_object->correlation_table,CL_TRUE,0,sizeof(int) * size_image, correlation_table, NULL, device_object->memory_copy_device_a);
    device_object->queue->enqueueWriteBuffer(*device_object->gain_correlation_map,CL_TRUE,0,sizeof(int) * size_image, gain_correlation_map, NULL, device_object->memory_copy_device_b);
    device_object->queue->enqueueWriteBuffer(*device_object->bad_pixel_map,CL_TRUE,0,sizeof(bool) * size_image, bad_pixel_map, NULL, device_object->memory_copy_device_c);
   
}
void copy_frame_to_device(DeviceObject *device_object, int* input_data, unsigned int size_image, unsigned int frame){
    device_object->queue->enqueueWriteBuffer(*device_object->image_input,CL_TRUE,0,sizeof(int) * size_image, input_data + (frame * size_image), NULL, NULL);
}

void process_full_frame_list (DeviceObject *device_object,int* input_frames,unsigned int frames, unsigned int size_frame,unsigned int w_size, unsigned int h_size){
   clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start);
    for (unsigned int frame = 0; frame < frames; ++frame )
    {
        // copy image
        copy_frame_to_device(device_object, input_frames, size_frame, frame);
        // process image
        process_image(device_object, w_size, h_size, frame);

    }
    device_object->queue->finish();
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end);
}

void process_image(DeviceObject *device_object, unsigned int w_size, unsigned int h_size, unsigned int frame){
    unsigned int size_image = w_size * h_size;
    // kernel 1
    // image offset correlation Gain correction
    unsigned int x_local= BLOCK_SIZE_PLANE;
    cl::NDRange local;
    cl::NDRange global;
    if ((w_size * h_size)  < BLOCK_SIZE_PLANE)
    {   
        printf("Entro 1");
        local = cl::NullRange;
        global = cl::NDRange(size_image);
    }
    else
    {
        local = cl::NDRange(x_local);
        global = cl::NDRange(size_image);
    }

    cl::Kernel kernel_image_offset_correlation=cl::Kernel(*device_object->program,"image_offset_correlation_gain_correction");
    kernel_image_offset_correlation.setArg(0,*device_object->image_input);
    kernel_image_offset_correlation.setArg(1,*device_object->correlation_table);
    kernel_image_offset_correlation.setArg(2,*device_object->gain_correlation_map);
    kernel_image_offset_correlation.setArg(3,*device_object->processing_image);
    kernel_image_offset_correlation.setArg(4,size_image);
    device_object->queue->enqueueNDRangeKernel(kernel_image_offset_correlation,cl::NullRange,global,local, NULL, NULL);

    // kernel 2
    // Bad pixel correction
    x_local= BLOCK_SIZE;
    unsigned int y_local = BLOCK_SIZE;
    if ( w_size < BLOCK_SIZE && h_size < BLOCK_SIZE)
    {
        printf("Entro 2");
        local = cl::NullRange;
        global = cl::NDRange(w_size, h_size);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(w_size, h_size);
    }

    cl::Kernel kernel_bad_pixel=cl::Kernel(*device_object->program,"bad_pixel_correlation");
    kernel_bad_pixel.setArg(0,*device_object->processing_image);
    kernel_bad_pixel.setArg(1,*device_object->processing_image_error_free);
    kernel_bad_pixel.setArg(2,*device_object->bad_pixel_map);
    kernel_bad_pixel.setArg(3,w_size);
    kernel_bad_pixel.setArg(4,h_size);
    device_object->queue->enqueueNDRangeKernel(kernel_bad_pixel,cl::NullRange,global,local, NULL, NULL);
    // kernel 3
    // spatial Binning Temporal Binning
    if ( w_size/2 < BLOCK_SIZE && h_size/2 < BLOCK_SIZE)
    {
        printf("Entro 3");
        local = cl::NullRange;
        global = cl::NDRange(w_size/2, h_size/2);
    }
    else
    {
        local = cl::NDRange(x_local, y_local);
        global = cl::NDRange(w_size/2, h_size/2);
    }

    cl::Kernel kernel_spatial_binning=cl::Kernel(*device_object->program,"spatial_binning_temporal_binning");
    kernel_spatial_binning.setArg(0,*device_object->processing_image_error_free);
    kernel_spatial_binning.setArg(1,*device_object->image_output);
    kernel_spatial_binning.setArg(2,w_size/2);
    kernel_spatial_binning.setArg(3,h_size/2);
    device_object->queue->enqueueNDRangeKernel(kernel_spatial_binning,cl::NullRange,global,local, NULL, NULL);
    // end kernels
}


void copy_memory_to_host(DeviceObject *device_object, int* output_image, unsigned int size_image){
    device_object->queue->enqueueReadBuffer(*device_object->image_output,CL_TRUE,0,size_image * sizeof(int),output_image, NULL, device_object->memory_copy_host);

}

void get_elapsed_time(DeviceObject *device_object, bool csv_format){
    device_object->memory_copy_host->wait();
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
    }
}



void clean(DeviceObject *device_object){
    // FINISH CLEAN STUFF
}
