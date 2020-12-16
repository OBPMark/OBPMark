#include <iostream>
#include <string>
#include <cstring>
#include <CL/cl.hpp>

#include "lib_functions.h"
#include "Config.h"
#include "GEN_kernel.hcl"



#define x_min 0
#define x_max pow( 2,n_bits) - 1

cl::CommandQueue queues[NUMBER_STREAMS][2 + n_bits];
void copy_data_to_cpu_asycronous(DataObject *device_object, int step);

void init(DataObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}

void init(DataObject *device_object, int platform ,int device, char* device_name){
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
    device_object->default_device = default_device;
    
    // events
    device_object->memory_copy_device = new cl::Event; 
    device_object->memory_copy_host_1 = new cl::Event;
    device_object->memory_copy_host_2 = new cl::Event;
    device_object->memory_copy_host_3 = new cl::Event;

    cl::Program::Sources sources;
    // load kernel from file
    std::string defines = "#define J_BlockSize " + std::to_string(J_BlockSize) +"\n" + "#define n_bits " + std::to_string(n_bits) +"\n";
    kernel_code = defines + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    // build
    device_object->program = new cl::Program(*device_object->context,sources);
    if(device_object->program->build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<device_object->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
}

bool device_memory_init(struct DataObject *device_object)
{   
    device_object->input_data = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof( unsigned long int ) * device_object->TotalSamplesStep);
    // data post_processed
    device_object->input_data_post_process = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof( unsigned long int ) * device_object->TotalSamples * NUMBER_STREAMS);

    device_object->missing_value = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * NUMBER_STREAMS);

    device_object->missing_value_inverse = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * NUMBER_STREAMS);

    device_object->zero_block_list = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * r_samplesInterval *  NUMBER_STREAMS);

    device_object->zero_block_list_inverse = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * r_samplesInterval *  NUMBER_STREAMS);

    // Aee variables 
    device_object->size_block = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned int) * r_samplesInterval *  NUMBER_STREAMS * (2 + n_bits));

    device_object->compresion_identifier = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned char) * r_samplesInterval *  NUMBER_STREAMS * (2 + n_bits));

    device_object->data_in_blocks_best = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned long int)  * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS);

    device_object->data_in_blocks_best_post_process = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned long int)  * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS );

    device_object->size_block_best = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned int) * r_samplesInterval *  NUMBER_STREAMS);

    device_object->compresion_identifier_best = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned char) * r_samplesInterval *  NUMBER_STREAMS);

    device_object->data_in_blocks = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(unsigned long int) * r_samplesInterval * J_BlockSize *  NUMBER_STREAMS * (2 + n_bits));


    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        for(unsigned int y = 0; y < 2 + n_bits; ++ y){
            queues[i][y] = cl::CommandQueue(*device_object->context,device_object->default_device,CL_QUEUE_PROFILING_ENABLE);
        }
    }

    // int CPU part
    device_object->data_in_blocks_best_cpu = ( unsigned long int *)malloc(sizeof( unsigned long int ) * device_object->TotalSamplesStep);
    device_object->size_block_best_cpu = ( unsigned int *)malloc(sizeof(unsigned int) * r_samplesInterval * STEPS);
    device_object->compresion_identifier_best_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * r_samplesInterval * STEPS);
    return true;
}

void copy_data_to_gpu(DataObject *device_object)
{
    queues[0][0].enqueueWriteBuffer(*device_object->input_data,CL_TRUE,0,sizeof( unsigned long int ) * device_object->TotalSamplesStep, device_object->InputDataBlock, NULL, device_object->memory_copy_device);
}

void execute_benchmark(struct DataObject *device_object)
{
    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;

    cl::NDRange local;
    cl::NDRange global;

    copy_data_to_gpu(device_object);
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_app);

    // Repeating the operations n times
    for(int step = 0; step < STEPS; ++step)
    {
        
        if (r_samplesInterval <= BLOCK_SIZE*BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(r_samplesInterval);
        }
        else
        {
            local = cl::NDRange(x_local * y_local);
            global = cl::NDRange(r_samplesInterval);
        }

        #ifdef PREPROCESSOR_ACTIVE
        // Preprocesor active
        cl::Kernel kernel_preprocessor=cl::Kernel(*device_object->program,"process_input_preprocessor");
        #else
        // Not preprocessor
        cl::Kernel kernel_preprocessor=cl::Kernel(*device_object->program,"process_input_no_preprocessor");  
        #endif
        kernel_preprocessor.setArg(0,*device_object->input_data);
        kernel_preprocessor.setArg(1,device_object->TotalSamples * step);
        kernel_preprocessor.setArg(2,*device_object->input_data_post_process);
        kernel_preprocessor.setArg(3,device_object->TotalSamples * (step % NUMBER_STREAMS));
        kernel_preprocessor.setArg(4,*device_object->zero_block_list);
        kernel_preprocessor.setArg(5,*device_object->zero_block_list_inverse);
        kernel_preprocessor.setArg(6,r_samplesInterval * (step % NUMBER_STREAMS));
        kernel_preprocessor.setArg(7,J_BlockSize);
        kernel_preprocessor.setArg(8,r_samplesInterval);

        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_preprocessor,cl::NullRange,global,local, NULL, NULL);
        // copy data and detect the zero blockls
        
        if (r_samplesInterval/2 <= BLOCK_SIZE*BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(r_samplesInterval/2);
        }
        else
        {
            local = cl::NDRange(x_local * y_local);
            global = cl::NDRange(r_samplesInterval/2);
        }

        cl::Kernel kernel_zero_block_fill=cl::Kernel(*device_object->program,"zero_block_list_completition");  
        kernel_zero_block_fill.setArg(0,*device_object->zero_block_list);
        kernel_zero_block_fill.setArg(1,*device_object->zero_block_list_inverse);
        kernel_zero_block_fill.setArg(2,r_samplesInterval * (step % NUMBER_STREAMS));
        kernel_zero_block_fill.setArg(3,*device_object->missing_value);
        kernel_zero_block_fill.setArg(4,*device_object->missing_value_inverse);
        kernel_zero_block_fill.setArg(5,(step % NUMBER_STREAMS));
        kernel_zero_block_fill.setArg(6,J_BlockSize);
        kernel_zero_block_fill.setArg(7,r_samplesInterval);

        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_zero_block_fill,cl::NullRange,global,local, NULL, NULL);
        // zero block finish 
        // start  adaptative entropy encoder
        // sync stream
        queues[step % NUMBER_STREAMS][0].finish();
        // processing of each block for aee
        // launch no_compresion and 0 block
        if (r_samplesInterval <= BLOCK_SIZE*BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(r_samplesInterval);
        }
        else
        {
            local = cl::NDRange(x_local * y_local);
            global = cl::NDRange(r_samplesInterval);
        }
        cl::Kernel aee_no_compresion=cl::Kernel(*device_object->program,"adaptative_entropy_encoder_no_compresion");  
        aee_no_compresion.setArg(0,*device_object->input_data_post_process);
        aee_no_compresion.setArg(1,device_object->TotalSamples * (step % NUMBER_STREAMS));
        aee_no_compresion.setArg(2,*device_object->zero_block_list);
        aee_no_compresion.setArg(3,*device_object->zero_block_list_inverse);
        aee_no_compresion.setArg(4,r_samplesInterval * (step % NUMBER_STREAMS));
        aee_no_compresion.setArg(5,*device_object->data_in_blocks);
        aee_no_compresion.setArg(6,r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits));
        aee_no_compresion.setArg(7,*device_object->size_block);
        aee_no_compresion.setArg(8,*device_object->compresion_identifier);
        aee_no_compresion.setArg(9,r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits));
        aee_no_compresion.setArg(10,0);
        aee_no_compresion.setArg(11,J_BlockSize);
        aee_no_compresion.setArg(12,r_samplesInterval);
        aee_no_compresion.setArg(13,n_bits);

        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(aee_no_compresion,cl::NullRange,global,local, NULL, NULL);
        
        // launch second extension
        cl::Kernel aee_second_extension=cl::Kernel(*device_object->program,"adaptative_entropy_encoder_second_extension");  
        aee_second_extension.setArg(0,*device_object->input_data_post_process);
        aee_second_extension.setArg(1,device_object->TotalSamples * (step % NUMBER_STREAMS));
        aee_second_extension.setArg(2,*device_object->zero_block_list);
        aee_second_extension.setArg(3,r_samplesInterval * (step % NUMBER_STREAMS));
        aee_second_extension.setArg(4,*device_object->data_in_blocks);
        aee_second_extension.setArg(5,r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits));
        aee_second_extension.setArg(6,*device_object->size_block);
        aee_second_extension.setArg(7,*device_object->compresion_identifier);
        aee_second_extension.setArg(8,r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits));
        aee_second_extension.setArg(9,1);
        aee_second_extension.setArg(10,J_BlockSize);
        aee_second_extension.setArg(11,r_samplesInterval);
        aee_second_extension.setArg(12,n_bits);

        queues[step % NUMBER_STREAMS][1].enqueueNDRangeKernel(aee_second_extension,cl::NullRange,global,local, NULL, NULL);
        
        // launch sample spiting
        for (unsigned int bit = 0; bit < n_bits; ++ bit)
        {
            cl::Kernel aee_sample_spliting=cl::Kernel(*device_object->program,"adaptative_entropy_encoder_sample_spliting");  
            aee_sample_spliting.setArg(0,*device_object->input_data_post_process);
            aee_sample_spliting.setArg(1,device_object->TotalSamples * (step % NUMBER_STREAMS));
            aee_sample_spliting.setArg(2,*device_object->zero_block_list);
            aee_sample_spliting.setArg(3,r_samplesInterval * (step % NUMBER_STREAMS));
            aee_sample_spliting.setArg(4,*device_object->data_in_blocks);
            aee_sample_spliting.setArg(5,r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits));
            aee_sample_spliting.setArg(6,*device_object->size_block);
            aee_sample_spliting.setArg(7,*device_object->compresion_identifier);
            aee_sample_spliting.setArg(8,r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits));
            aee_sample_spliting.setArg(9,bit + 2);
            aee_sample_spliting.setArg(10,J_BlockSize);
            aee_sample_spliting.setArg(11,r_samplesInterval);
            aee_sample_spliting.setArg(12,n_bits);

            queues[step % NUMBER_STREAMS][bit + 2].enqueueNDRangeKernel(aee_sample_spliting,cl::NullRange,global,local, NULL, NULL);
            

        }
        // sync screams of the same "primary" stream
        for(unsigned int y = 0; y < 2 + n_bits; ++ y)
        {
            queues[step % NUMBER_STREAMS][y].finish();
        }
        // block selector
        cl::Kernel aee_block_selector=cl::Kernel(*device_object->program,"adaptative_entropy_encoder_block_selector");  
        aee_block_selector.setArg(0,*device_object->zero_block_list);
        aee_block_selector.setArg(1,r_samplesInterval * (step % NUMBER_STREAMS));
        aee_block_selector.setArg(2,*device_object->data_in_blocks);
        aee_block_selector.setArg(3,r_samplesInterval * J_BlockSize  * (step % NUMBER_STREAMS) * (2 + n_bits));
        aee_block_selector.setArg(4,*device_object->size_block);
        aee_block_selector.setArg(5,*device_object->compresion_identifier);
        aee_block_selector.setArg(6,(r_samplesInterval * (step % NUMBER_STREAMS) * (2 + n_bits)));
        aee_block_selector.setArg(7,*device_object->data_in_blocks_best);
        aee_block_selector.setArg(8,J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS));
        aee_block_selector.setArg(9,*device_object->size_block_best);
        aee_block_selector.setArg(10,*device_object->compresion_identifier_best);
        aee_block_selector.setArg(11,r_samplesInterval * (step % NUMBER_STREAMS));
        aee_block_selector.setArg(12,J_BlockSize);
        aee_block_selector.setArg(13,r_samplesInterval);
        aee_block_selector.setArg(14,n_bits);
        
        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(aee_block_selector,cl::NullRange,global,local, NULL, NULL);
        // precompute the data

        if (r_samplesInterval <= BLOCK_SIZE || 32 * J_BlockSize <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(r_samplesInterval, 32 * J_BlockSize);
        }
        else
        {
            local = cl::NDRange(x_local , y_local);
            global = cl::NDRange(r_samplesInterval, 32 * J_BlockSize);
        }
        cl::Kernel kernel_post_process=cl::Kernel(*device_object->program,"post_processing_of_output_data");
        kernel_post_process.setArg(0,*device_object->data_in_blocks_best);
        kernel_post_process.setArg(1,J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS));
        kernel_post_process.setArg(2,*device_object->size_block_best);
        kernel_post_process.setArg(3,*device_object->compresion_identifier_best);
        kernel_post_process.setArg(4,r_samplesInterval * (step % NUMBER_STREAMS));
        kernel_post_process.setArg(5,*device_object->data_in_blocks_best_post_process);
        kernel_post_process.setArg(6,J_BlockSize * r_samplesInterval * (step % NUMBER_STREAMS));
        kernel_post_process.setArg(7,J_BlockSize);
        kernel_post_process.setArg(8,r_samplesInterval);

        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_post_process,cl::NullRange,global,local, NULL, NULL);
        // SYNC ALL STREAMS
        for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
                queues[i][0].finish();
        }
        if (step % NUMBER_STREAMS == 0)
        {
            copy_data_to_cpu_asycronous(device_object, step);
        }
        
    }
    // sync GPU
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        for(unsigned int y = 0; y < 2 + n_bits; ++ y){
            queues[i][y].finish();
        }
    }
    // GPU SYNC
    //printf("\n\n\n\n");
    /*for(unsigned int x = 0; x < r_samplesInterval; ++x)
    {
        printf("%u %u %lu|", device_object->size_block_best_cpu[x], device_object->compresion_identifier_best_cpu[x], device_object->data_in_blocks_best_cpu[x * J_BlockSize]);
    }
    for(unsigned int x = 0; x < r_samplesInterval; ++x)
    {
        printf("Size: %u, id: %u, data: %lu\n", device_object->size_block_best_cpu[x], (unsigned int)device_object->compresion_identifier_best_cpu[x],device_object->data_in_blocks_best_cpu[x*J_BlockSize]);
    }*/

    unsigned int acc_size = 0;
    for(int step = 0; step < STEPS; ++step)
    {    
        // header
        for(unsigned char bit = 0; bit < 6; ++bit)
        {
            if((n_bits & (1 << (bit%8) )) != 0)
            {
                device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
            }
        }
        acc_size += 6;
        for(unsigned int block = 0; block < r_samplesInterval; ++block)
        {  

            // reprocess the data
            
            for(int bit = 0; bit < device_object->size_block_best_cpu[block + (step * r_samplesInterval)]; ++bit)
            {
                
                if((device_object->data_in_blocks_best_cpu[(bit/32) + (block * J_BlockSize) + (step * r_samplesInterval * J_BlockSize)] & (1 << (bit%32) )) != 0)
                {
                    device_object->OutputDataBlock[(acc_size+bit)/32] |= 1 << ((acc_size+bit)%32);
                }
            }
            acc_size += device_object->size_block_best_cpu[block + (step * r_samplesInterval)];
            
        
        }
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_app);
    //cudaEventRecord(*device_object->stop_app);
    // Debug loop - uncomment if needed
    //printf("%d\n\n", acc_size);
    for(int i = acc_size - 1; i >= 0; --i)
    {
        printf("%d" ,(device_object->OutputDataBlock[i/32] & (1 << (i%32) )) != 0);
    }
    printf("\n");

}

void copy_data_to_cpu_asycronous(DataObject *device_object, int step)
{

queues[step % NUMBER_STREAMS][0].enqueueReadBuffer(*device_object->data_in_blocks_best_post_process,CL_FALSE,0,sizeof( unsigned long int ) * r_samplesInterval * J_BlockSize  * NUMBER_STREAMS,device_object->data_in_blocks_best_cpu + (J_BlockSize * r_samplesInterval * step), NULL, device_object->memory_copy_host_1);
queues[step % NUMBER_STREAMS][1].enqueueReadBuffer(*device_object->size_block_best ,CL_FALSE,0,sizeof( unsigned int ) * r_samplesInterval * NUMBER_STREAMS,device_object->size_block_best_cpu + (r_samplesInterval * step), NULL, device_object->memory_copy_host_2);
queues[step % NUMBER_STREAMS][2].enqueueReadBuffer(*device_object->compresion_identifier_best ,CL_FALSE,0,sizeof( unsigned char ) * r_samplesInterval  * NUMBER_STREAMS,device_object->compresion_identifier_best_cpu + (r_samplesInterval * step), NULL, device_object->memory_copy_host_3);
    
}

void get_elapsed_time(DataObject *device_object, bool csv_format){
    //cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float elapsed_h_d = 0, elapsed_d_h = 0;
    // memory transfer time host-device
    elapsed_h_d =  device_object->memory_copy_device->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_device->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    // Total Time
    long unsigned int application_miliseconds = (device_object->end_app.tv_sec - device_object->start_app.tv_sec) * 1000 + (device_object->end_app.tv_nsec - device_object->start_app.tv_nsec) / 1000000;
    //  memory transfer time device-host
    elapsed_d_h =  device_object->memory_copy_host_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_host_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h +=device_object->memory_copy_host_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_host_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_d_h +=device_object->memory_copy_host_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->memory_copy_host_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    
    if (csv_format){
         printf("%.10f;%lu;%.10f;\n", elapsed_h_d/ 1000000.0,application_miliseconds,elapsed_d_h/ 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", elapsed_h_d/ 1000000.0);
         printf("Elapsed time kernel: %lu miliseconds\n", application_miliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h/ 1000000.0);
    }

}

void clean(struct DataObject *device_object)
{
    /*free(device_object->InputDataBlock);
    free(device_object->OutputDataBlock);
    free(device_object);*/
}