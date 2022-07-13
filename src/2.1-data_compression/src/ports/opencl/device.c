/**
 * \file device.c
 * \brief Benchmark #121 OpenCL version device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "device.h"
#include "GEN_processing.hcl"
 

 void init(
     compression_data_t *compression_data,
     compression_time_t *t,
     char *device_name
     )
 {
     init(compression_data,t, 0,0, device_name);
 }
 
 
 
 void init(
     compression_data_t *compression_data,
     compression_time_t *t,
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
    cl::Device* default_device= new cl::Device (all_devices[device]);
    //std::cout<< "Using device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<"\n";
    strcpy(device_name,default_device->getInfo<CL_DEVICE_NAME>().c_str() );
    // context
    compression_data->default_device = default_device;
    compression_data->context = new cl::Context(*compression_data->default_device);
    compression_data->queue = new cl::CommandQueue(*compression_data->context,*compression_data->default_device,NULL); //CL_QUEUE_PROFILING_ENABLE
    
    
    //event create 
    t->t_host_device = new cl::Event();
    t->t_device_host_1 = new cl::Event();
    t->t_device_host_2 = new cl::Event();
    t->t_device_host_3 = new cl::Event();
    t->t_device_host_4 = new cl::Event();
    
    
    // program
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_def_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    compression_data->program = new cl::Program(*compression_data->context,sources);
    if(compression_data->program->build({*compression_data->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<compression_data->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*compression_data->default_device)<<"\n";
        exit(1);
    }
}


bool device_memory_init(
	compression_data_t *compression_data
	)
{	
	
    compression_data->input_data = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof( unsigned int  ) * compression_data->TotalSamplesStep);
    
    // data post_processed
    compression_data->input_data_post_process = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof( unsigned int  ) * compression_data->TotalSamples * NUMBER_STREAMS);
    // data out
    compression_data->output_data = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof( unsigned int  ) * compression_data->TotalSamplesStep);
   
    compression_data->missing_value = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(int) * NUMBER_STREAMS);
    compression_data->missing_value_inverse = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(int) * NUMBER_STREAMS);

    compression_data->zero_block_list = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->zero_block_list_status = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->zero_block_list_inverse = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);

    // Aee variables 
    compression_data->size_block = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    compression_data->compresion_identifier = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    compression_data->compresion_identifier_internal = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS * (2 + compression_data->n_bits));
    
    compression_data->data_in_blocks_best = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned int )  * compression_data->r_samplesInterval * compression_data->j_blocksize *  NUMBER_STREAMS);
    compression_data->size_block_best = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->bit_block_best = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned int) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->compresion_identifier_best = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->compresion_identifier_internal_best = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned char) * compression_data->r_samplesInterval *  NUMBER_STREAMS);
    compression_data->data_in_blocks = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE , sizeof(unsigned int ) * compression_data->r_samplesInterval * compression_data->j_blocksize *  NUMBER_STREAMS * (2 + compression_data->n_bits));

    // int CPU part
    compression_data->data_in_blocks_best_cpu = ( unsigned int *)malloc(sizeof( unsigned int ) * compression_data->TotalSamplesStep);
    compression_data->size_block_best_cpu = ( unsigned int *)malloc(sizeof(unsigned int) * compression_data->r_samplesInterval * compression_data->steps);
    compression_data->compresion_identifier_best_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * compression_data->r_samplesInterval * compression_data->steps);
    compression_data->compresion_identifier_best_internal_cpu = ( unsigned char *)malloc(sizeof( unsigned char) * compression_data->r_samplesInterval * compression_data->steps);
    
	return true;
}



void copy_memory_to_device(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
    compression_data->queue->enqueueWriteBuffer(*compression_data->input_data, CL_TRUE,0,sizeof( unsigned int  ) * compression_data->TotalSamplesStep,compression_data->InputDataBlock,NULL,t->t_host_device);
}


void copy_data_to_cpu_asynchronous(
    compression_data_t *compression_data,
	compression_time_t *t, 
    int step)
{


    cl_buffer_region data_in_blocks = {((step % NUMBER_STREAMS) * compression_data->j_blocksize * compression_data->r_samplesInterval * sizeof( unsigned  int )),  (((step % NUMBER_STREAMS) * compression_data->j_blocksize * compression_data->r_samplesInterval) +  compression_data->r_samplesInterval * compression_data->j_blocksize) * sizeof( unsigned  int )};
    cl::Buffer data_in_blocks_best_section = compression_data->data_in_blocks_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &data_in_blocks);
    
    compression_data->queue->enqueueReadBuffer(data_in_blocks_best_section,CL_TRUE,0,sizeof( unsigned  int ) * compression_data->r_samplesInterval * compression_data->j_blocksize, compression_data->data_in_blocks_best_cpu + (compression_data->j_blocksize * compression_data->r_samplesInterval * step), NULL, t->t_device_host_1);
    
    cl_buffer_region size_blocks = {((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned int )),  (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) + compression_data->r_samplesInterval) * sizeof( unsigned int )  };
    cl::Buffer size_blocks_section = compression_data->size_block_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &size_blocks);
    
    compression_data->queue->enqueueReadBuffer(size_blocks_section,CL_TRUE,0,sizeof( unsigned int ) * compression_data->r_samplesInterval, compression_data->size_block_best_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_2);

    cl_buffer_region compression_identifier = { ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned char )),   (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) +  compression_data->r_samplesInterval) *sizeof( unsigned char )};
    cl::Buffer compression_identifier_section = compression_data->compresion_identifier_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compression_identifier);
    
    compression_data->queue->enqueueReadBuffer(compression_identifier_section,CL_TRUE,0,sizeof( unsigned char ) * compression_data->r_samplesInterval, compression_data->compresion_identifier_best_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_3);


    cl_buffer_region compression_identifier_internal = { ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned char )),   (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) +  compression_data->r_samplesInterval) * sizeof( unsigned char )};
    cl::Buffer compression_identifier_internal_section = compression_data->compresion_identifier_internal_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compression_identifier_internal);
    
    compression_data->queue->enqueueReadBuffer(compression_identifier_internal_section,CL_TRUE,0,sizeof( unsigned char ) * compression_data->r_samplesInterval, compression_data->compresion_identifier_best_internal_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_4);



}

void process_benchmark(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
   
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int x_local_plane= BLOCK_SIZE_PLANE;
    const unsigned int y_local= BLOCK_SIZE;

    T_START(t->t_test);

    cl::CommandQueue queues[NUMBER_STREAMS][MAXSIZE_NBITS];
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        for (unsigned int j = 0; j < MAXSIZE_NBITS; ++j) {
            queues[i][j] = cl::CommandQueue(*compression_data->context,*compression_data->default_device,NULL);//CL_QUEUE_PROFILING_ENABLE
        }
         
    }

    // Repeating the operations n times
    for(int step = 0; step < compression_data->steps; ++step)
    {
        // check if preprocessing is required
        
    
        cl::NDRange local_prepo;
        cl::NDRange global_prepo;

        if (compression_data->r_samplesInterval < BLOCK_SIZE_PLANE)
        {
            local_prepo = cl::NullRange;
            global_prepo = cl::NDRange(compression_data->r_samplesInterval);
        }
        else
        {
            local_prepo = cl::NDRange(x_local_plane);
            global_prepo = cl::NDRange(compression_data->r_samplesInterval);
        }
        
        
        /*cl_buffer_region input_data_internal = { (compression_data->TotalSamples * step * sizeof( unsigned int )),   ((compression_data->TotalSamples * step ) + compression_data->r_samplesInterval)  * sizeof( unsigned int )};
        cl::Buffer input_data_internal_section = compression_data->input_data->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &input_data_internal);
        
        cl_buffer_region input_data_post_process = { (compression_data->TotalSamples * (step % NUMBER_STREAMS) * sizeof( unsigned int )),   ((compression_data->TotalSamples * (step % NUMBER_STREAMS)  ) + compression_data->r_samplesInterval )* sizeof( unsigned int )};
        cl::Buffer input_data_post_process_section = compression_data->input_data_post_process->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &input_data_post_process);

        cl_buffer_region zero_block_list = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * sizeof(  int )),   ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS)  ) + compression_data->r_samplesInterval ) * sizeof(  int )};
        cl::Buffer zero_block_list_section = compression_data->zero_block_list->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &zero_block_list);

        cl_buffer_region zero_block_list_inverse = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * sizeof(  int )),   ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS)  ) + compression_data->r_samplesInterval) * sizeof(  int )};
        cl::Buffer zero_block_list_inverse_section = compression_data->zero_block_list_inverse->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &zero_block_list_inverse);

        cl_buffer_region data_in_blocks = {  (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits) * sizeof( unsigned int )),   ((compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) + compression_data->r_samplesInterval * compression_data->j_blocksize) * sizeof( unsigned int )};
        cl::Buffer data_in_blocks_section = compression_data->data_in_blocks->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &data_in_blocks);
        
        cl_buffer_region size_block = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) * sizeof( unsigned int ),  ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) + compression_data->r_samplesInterval) * sizeof( unsigned int )};
        cl::Buffer size_block_section = compression_data->size_block->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &size_block);
        

        cl_buffer_region compresion_identifier = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) * sizeof( unsigned char ),  ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) + compression_data->r_samplesInterval) * sizeof( unsigned char )};
        cl::Buffer compresion_identifier_section = compression_data->compresion_identifier->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compresion_identifier);
        
        cl_buffer_region compresion_identifier_internal = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) * sizeof( unsigned char ),  ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits)) + compression_data->r_samplesInterval) * sizeof( unsigned char )};
        cl::Buffer compresion_identifier_internal_section = compression_data->compresion_identifier_internal->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compresion_identifier_internal);
         
        cl_buffer_region halved_samples = {  (compression_data->j_blocksize)/2 * (step % NUMBER_STREAMS) * sizeof( unsigned int ),  ((compression_data->j_blocksize)/2 * (step % NUMBER_STREAMS) + compression_data->j_blocksize/2 ) * sizeof( unsigned int )};
        cl::Buffer halved_samples_section = compression_data->halved_samples->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &halved_samples);*/
        unsigned int offset_input_data_internal =  (compression_data->TotalSamples * step);
        unsigned int offset_input_data_post_process =  (compression_data->TotalSamples * (step % NUMBER_STREAMS));
        unsigned int offset_zero_block_list =  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS));


        if(compression_data->preprocessor_active)
        {   
            cl::Kernel kernel_prepro=cl::Kernel(*compression_data->program,"process_input_preprocessor");
                        kernel_prepro.setArg(0, *compression_data->input_data );
                        kernel_prepro.setArg(1, *compression_data->input_data_post_process);
                        kernel_prepro.setArg(2, *compression_data->zero_block_list_status);
                        kernel_prepro.setArg(3, *compression_data->zero_block_list);
                        kernel_prepro.setArg(4, *compression_data->zero_block_list_inverse);
                        kernel_prepro.setArg(5, compression_data->j_blocksize);
                        kernel_prepro.setArg(6, compression_data->r_samplesInterval);
                        kernel_prepro.setArg(7, compression_data->n_bits);
                        kernel_prepro.setArg(8, (unsigned int)(pow( 2,compression_data->n_bits) - 1));
                        kernel_prepro.setArg(9, offset_input_data_internal);
                        kernel_prepro.setArg(10, offset_input_data_post_process);
                        kernel_prepro.setArg(11, offset_zero_block_list);

           queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_prepro,cl::NullRange,global_prepo,local_prepo, NULL, NULL);
        }
        else
        {
            cl::Kernel kernel_no_prepro=cl::Kernel(*compression_data->program,"process_input_no_preprocessor");
                        kernel_no_prepro.setArg(0, *compression_data->input_data);
                        kernel_no_prepro.setArg(1, *compression_data->input_data_post_process);
                        kernel_no_prepro.setArg(2, *compression_data->zero_block_list_status);
                        kernel_no_prepro.setArg(3, *compression_data->zero_block_list);
                        kernel_no_prepro.setArg(4, *compression_data->zero_block_list_inverse);
                        kernel_no_prepro.setArg(5, compression_data->j_blocksize);
                        kernel_no_prepro.setArg(6, compression_data->r_samplesInterval);
                        kernel_no_prepro.setArg(7, offset_input_data_internal);
                        kernel_no_prepro.setArg(8, offset_input_data_post_process);
                        kernel_no_prepro.setArg(9, offset_zero_block_list);
            queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_no_prepro,cl::NullRange,global_prepo,local_prepo, NULL, NULL);

        }

        
        cl::NDRange local_zero;
        cl::NDRange global_zero;

        if ((compression_data->r_samplesInterval)/2 < BLOCK_SIZE_PLANE)
        {
            local_zero = cl::NullRange;
            global_zero = cl::NDRange((compression_data->r_samplesInterval)/2);
        }
        else
        {
            local_zero = cl::NDRange(x_local_plane);
            global_zero = cl::NDRange((compression_data->r_samplesInterval)/2);
        }


        cl::Kernel kernel_zero=cl::Kernel(*compression_data->program,"zero_block_list_completition");
                    kernel_zero.setArg(0, *compression_data->zero_block_list);
                    kernel_zero.setArg(1, *compression_data->zero_block_list_inverse);
                    kernel_zero.setArg(2, *compression_data->missing_value);
                    kernel_zero.setArg(3, *compression_data->missing_value_inverse);
                    kernel_zero.setArg(4, (int)(step % NUMBER_STREAMS));
                    kernel_zero.setArg(5, compression_data->j_blocksize);
                    kernel_zero.setArg(6, compression_data->r_samplesInterval);
                    kernel_zero.setArg(7, offset_zero_block_list);
        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_zero,cl::NullRange,global_zero,local_zero, NULL, NULL);



        // zero block finish 

        // sync stream
        queues[step % NUMBER_STREAMS][0].finish();

        // start  adaptative entropy encoder
        cl::NDRange local_adaptative;
        cl::NDRange global_adaptative;

        if (compression_data->r_samplesInterval < BLOCK_SIZE && compression_data->j_blocksize < BLOCK_SIZE)
        {
            local_adaptative = cl::NullRange;
            global_adaptative = cl::NDRange(compression_data->r_samplesInterval, compression_data->j_blocksize);
        }
        else
        {
            local_adaptative = cl::NDRange(x_local, y_local);
            global_adaptative = cl::NDRange(compression_data->r_samplesInterval, compression_data->j_blocksize);
        }

        unsigned int offset_data_in_blocks = (compression_data->r_samplesInterval * compression_data->j_blocksize  * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits));
        unsigned int offset_size_block = (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits));
        unsigned int offset_compresion_identifier = (compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * (2 + compression_data->n_bits));

        cl::Kernel kernel_nocompression=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_no_compresion");
                    kernel_nocompression.setArg(0, *compression_data->input_data_post_process);
                    kernel_nocompression.setArg(1, *compression_data->zero_block_list_status);
                    kernel_nocompression.setArg(2, *compression_data->data_in_blocks);
                    kernel_nocompression.setArg(3, *compression_data->size_block);
                    kernel_nocompression.setArg(4, *compression_data->compresion_identifier);
                    kernel_nocompression.setArg(5, *compression_data->compresion_identifier_internal);
                    kernel_nocompression.setArg(6, 0);
                    kernel_nocompression.setArg(7, compression_data->j_blocksize);
                    kernel_nocompression.setArg(8, compression_data->r_samplesInterval);
                    kernel_nocompression.setArg(9, compression_data->n_bits);
                    kernel_nocompression.setArg(10, offset_input_data_post_process);
                    kernel_nocompression.setArg(11, offset_zero_block_list);
                    kernel_nocompression.setArg(12, offset_data_in_blocks);
                    kernel_nocompression.setArg(13, offset_size_block);
                    kernel_nocompression.setArg(14, offset_compresion_identifier);

        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_nocompression,cl::NullRange,global_adaptative,local_adaptative, NULL, NULL);

        cl::NDRange local_zero_block;
        cl::NDRange global_zero_block;

        if (compression_data->r_samplesInterval < BLOCK_SIZE_PLANE)
        {
            local_zero_block = cl::NullRange;
            global_zero_block = cl::NDRange(compression_data->r_samplesInterval);
        }
        else
        {
            local_zero_block = cl::NDRange(x_local);
            global_zero_block = cl::NDRange(compression_data->r_samplesInterval);
        }
    cl::Kernel kernel_zeroblock=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_zero_block");
                    kernel_zeroblock.setArg(0, *compression_data->input_data_post_process);
                    kernel_zeroblock.setArg(1, *compression_data->zero_block_list);
                    kernel_zeroblock.setArg(2, *compression_data->zero_block_list_inverse);
                    kernel_zeroblock.setArg(3, *compression_data->data_in_blocks);
                    kernel_zeroblock.setArg(4, *compression_data->size_block);
                    kernel_zeroblock.setArg(5, *compression_data->compresion_identifier);
                    kernel_zeroblock.setArg(6, *compression_data->compresion_identifier_internal);
                    kernel_zeroblock.setArg(7, 0);
                    kernel_zeroblock.setArg(8, compression_data->j_blocksize);
                    kernel_zeroblock.setArg(9, compression_data->r_samplesInterval);
                    kernel_zeroblock.setArg(10, compression_data->n_bits);
                    kernel_zeroblock.setArg(11, offset_input_data_post_process);
                    kernel_zeroblock.setArg(12, offset_zero_block_list);
                    kernel_zeroblock.setArg(13, offset_data_in_blocks);
                    kernel_zeroblock.setArg(14, offset_size_block);
                    kernel_zeroblock.setArg(15, offset_compresion_identifier);


        queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_zeroblock,cl::NullRange,global_zero_block,local_zero_block, NULL, NULL);
    

        // launch second extension
        cl::NDRange local_secondextension;
        cl::NDRange global_secondextension;

        if (compression_data->r_samplesInterval < BLOCK_SIZE_PLANE)
        {
            local_secondextension = cl::NullRange;
            global_secondextension = cl::NDRange(compression_data->r_samplesInterval);
        }
        else
        {
            local_secondextension = cl::NDRange(x_local_plane);
            global_secondextension = cl::NDRange(compression_data->r_samplesInterval);
        }


        cl::Kernel kernel_second=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_second_extension");
                    kernel_second.setArg(0, *compression_data->input_data_post_process);
                    kernel_second.setArg(1, *compression_data->zero_block_list_status);
                    kernel_second.setArg(2, *compression_data->data_in_blocks);
                    kernel_second.setArg(3, *compression_data->size_block);
                    kernel_second.setArg(4, *compression_data->compresion_identifier);
                    kernel_second.setArg(5, *compression_data->compresion_identifier_internal);
                    kernel_second.setArg(6, 1);
                    kernel_second.setArg(7, compression_data->j_blocksize);
                    kernel_second.setArg(8, compression_data->r_samplesInterval);
                    kernel_second.setArg(9, compression_data->n_bits);
                    kernel_second.setArg(10, offset_input_data_post_process);
                    kernel_second.setArg(11, offset_zero_block_list);
                    kernel_second.setArg(12, offset_data_in_blocks);
                    kernel_second.setArg(13, offset_size_block);
                    kernel_second.setArg(14, offset_compresion_identifier);

        queues[step % NUMBER_STREAMS][1].enqueueNDRangeKernel(kernel_second,cl::NullRange,global_secondextension,local_secondextension, NULL, NULL);

        
        cl::NDRange local_samplespliting;
        cl::NDRange global_samplespliting;

        if (compression_data->r_samplesInterval < BLOCK_SIZE_PLANE)
        {
            local_samplespliting = cl::NullRange;
            global_samplespliting = cl::NDRange(compression_data->r_samplesInterval);
        }
        else
        {
            local_samplespliting = cl::NDRange(x_local_plane);
            global_samplespliting = cl::NDRange(compression_data->r_samplesInterval);
        }
        
        // launch sample spiting
        for (unsigned int bit = 0; bit < compression_data->n_bits; ++ bit)
        {

            cl::Kernel kernel_sample=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_sample_spliting");
                        kernel_sample.setArg(0, *compression_data->input_data_post_process);
                        kernel_sample.setArg(1, *compression_data->zero_block_list_status);
                        kernel_sample.setArg(2, *compression_data->data_in_blocks);
                        kernel_sample.setArg(3, *compression_data->size_block);
                        kernel_sample.setArg(4, *compression_data->compresion_identifier);
                        kernel_sample.setArg(5, *compression_data->compresion_identifier_internal);
                        kernel_sample.setArg(6, bit + 2);
                        kernel_sample.setArg(7, compression_data->j_blocksize);
                        kernel_sample.setArg(8, compression_data->r_samplesInterval);
                        kernel_sample.setArg(9, compression_data->n_bits);
                        kernel_sample.setArg(10, offset_input_data_post_process);
                        kernel_sample.setArg(11, offset_zero_block_list);
                        kernel_sample.setArg(12, offset_data_in_blocks);
                        kernel_sample.setArg(13, offset_size_block);
                        kernel_sample.setArg(14, offset_compresion_identifier);

            queues[step % NUMBER_STREAMS][bit + 2].enqueueNDRangeKernel(kernel_sample,cl::NullRange,global_samplespliting,local_samplespliting, NULL, NULL);
        }
        
        // sync screams of the same "primary" stream
        for(unsigned int y = 0; y < 2 + compression_data->n_bits; ++ y)
        {
            queues[step % NUMBER_STREAMS][y].finish();
        }
       
        
        // block selector 

        cl::NDRange local_blockselector;
        cl::NDRange global_blockselector;

        if (compression_data->r_samplesInterval < BLOCK_SIZE_PLANE)
        {
            local_blockselector = cl::NullRange;
            global_blockselector = cl::NDRange(compression_data->r_samplesInterval);
        }
        else
        {
            local_blockselector = cl::NDRange(x_local_plane);
            global_blockselector = cl::NDRange(compression_data->r_samplesInterval);
        }
        
        
        /*cl_buffer_region bit_block_best = {  (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS)) * sizeof( unsigned int ),  ((compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS))  + compression_data->r_samplesInterval) * sizeof( unsigned int )};
        cl::Buffer bit_block_best_section = compression_data->bit_block_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &bit_block_best);
        
        cl_buffer_region size_block_best = {  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)) * sizeof( unsigned int ),  ((compression_data->r_samplesInterval * (step % NUMBER_STREAMS))  + compression_data->r_samplesInterval) * sizeof( unsigned int )};
        cl::Buffer size_block_best_section = compression_data->size_block_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &size_block_best);
        
        cl_buffer_region compresion_identifier_best = {  compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * sizeof( unsigned char ),  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)  + compression_data->r_samplesInterval) * sizeof( unsigned char )};
        cl::Buffer compresion_identifier_best_section = compression_data->compresion_identifier_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compresion_identifier_best);

        cl_buffer_region compresion_identifier_internal_best = {  compression_data->r_samplesInterval * (step % NUMBER_STREAMS) * sizeof( unsigned char ),  (compression_data->r_samplesInterval * (step % NUMBER_STREAMS)  + compression_data->r_samplesInterval) * sizeof( unsigned char )};
        cl::Buffer compresion_identifier_internal_best_section = compression_data->compresion_identifier_internal_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compresion_identifier_best);
        */

        unsigned int offset_bit_block_best = (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS));
        unsigned int offset_size_block_best = (compression_data->r_samplesInterval * (step % NUMBER_STREAMS));
        unsigned int offset_compresion_identifier_best = (compression_data->r_samplesInterval * (step % NUMBER_STREAMS));

        cl::Kernel kernel_blockselector=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_block_selector");
                        kernel_blockselector.setArg(0, *compression_data->zero_block_list_status);
                        kernel_blockselector.setArg(1, *compression_data->bit_block_best );
                        kernel_blockselector.setArg(2, *compression_data->size_block);
                        kernel_blockselector.setArg(3, *compression_data->compresion_identifier);
                        kernel_blockselector.setArg(4, *compression_data->compresion_identifier_internal);
                        kernel_blockselector.setArg(5, *compression_data->size_block_best);
                        kernel_blockselector.setArg(6, *compression_data->compresion_identifier_best);
                        kernel_blockselector.setArg(7, *compression_data->compresion_identifier_internal_best);
                        kernel_blockselector.setArg(8, compression_data->j_blocksize);
                        kernel_blockselector.setArg(9, compression_data->r_samplesInterval);
                        kernel_blockselector.setArg(10, compression_data->n_bits);
                        kernel_blockselector.setArg(11, offset_zero_block_list);
                        kernel_blockselector.setArg(12, offset_bit_block_best);
                        kernel_blockselector.setArg(13, offset_size_block);
                        kernel_blockselector.setArg(14, offset_compresion_identifier);
                        kernel_blockselector.setArg(15, offset_size_block_best);
                        kernel_blockselector.setArg(16, offset_compresion_identifier_best);
            queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_blockselector,cl::NullRange,global_blockselector,local_blockselector, NULL, NULL);
        

        cl::NDRange local_blockselectorcopy;
        cl::NDRange global_blockselectorcopy;

        if (compression_data->r_samplesInterval < BLOCK_SIZE && compression_data->j_blocksize < BLOCK_SIZE)
        {
            local_blockselectorcopy = cl::NullRange;
            global_blockselectorcopy = cl::NDRange(compression_data->r_samplesInterval, compression_data->j_blocksize);
        }
        else
        {
            local_blockselectorcopy = cl::NDRange(x_local, y_local);
            global_blockselectorcopy = cl::NDRange(compression_data->r_samplesInterval, compression_data->j_blocksize);
        }

        /*cl_buffer_region data_in_blocks_best = {  compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS),  compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS) + compression_data->j_blocksize * compression_data->r_samplesInterval * sizeof( unsigned char )};
        cl::Buffer data_in_blocks_best_section = compression_data->data_in_blocks_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compresion_identifier_best);*/
        unsigned int offset_data_in_blocks_best = (compression_data->j_blocksize * compression_data->r_samplesInterval * (step % NUMBER_STREAMS));


         cl::Kernel kernel_datacopy=cl::Kernel(*compression_data->program,"adaptative_entropy_encoder_block_selector_data_copy");
                        kernel_datacopy.setArg(0, *compression_data->zero_block_list);
                        kernel_datacopy.setArg(1, *compression_data->data_in_blocks);
                        kernel_datacopy.setArg(2, *compression_data->bit_block_best);
                        kernel_datacopy.setArg(3, *compression_data->data_in_blocks_best );
                        kernel_datacopy.setArg(4, compression_data->j_blocksize);
                        kernel_datacopy.setArg(5, compression_data->r_samplesInterval);
                        kernel_datacopy.setArg(6, offset_zero_block_list);
                        kernel_datacopy.setArg(7, offset_data_in_blocks);
                        kernel_datacopy.setArg(8, offset_bit_block_best);
                        kernel_datacopy.setArg(9, offset_data_in_blocks_best);
            queues[step % NUMBER_STREAMS][0].enqueueNDRangeKernel(kernel_datacopy,cl::NullRange,global_blockselectorcopy,local_blockselectorcopy, NULL, NULL);

        // copy back the data
        //copy_data_to_cpu_asynchronous(compression_data,t, step);

        cl_buffer_region data_in_blocks = {((step % NUMBER_STREAMS) * compression_data->j_blocksize * compression_data->r_samplesInterval * sizeof( unsigned  int )),  (((step % NUMBER_STREAMS) * compression_data->j_blocksize * compression_data->r_samplesInterval) +  compression_data->r_samplesInterval * compression_data->j_blocksize) * sizeof( unsigned  int )};
        cl::Buffer data_in_blocks_best_section = compression_data->data_in_blocks_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &data_in_blocks);
    
        queues[step % NUMBER_STREAMS][0].enqueueReadBuffer(data_in_blocks_best_section,CL_TRUE,0,sizeof( unsigned  int ) * compression_data->r_samplesInterval * compression_data->j_blocksize, compression_data->data_in_blocks_best_cpu + (compression_data->j_blocksize * compression_data->r_samplesInterval * step), NULL, t->t_device_host_1);
        
        cl_buffer_region size_blocks = {((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned int )),  (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) + compression_data->r_samplesInterval) * sizeof( unsigned int )  };
        cl::Buffer size_blocks_section = compression_data->size_block_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &size_blocks);
        
        queues[step % NUMBER_STREAMS][1].enqueueReadBuffer(size_blocks_section,CL_TRUE,0,sizeof( unsigned int ) * compression_data->r_samplesInterval, compression_data->size_block_best_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_2);

        cl_buffer_region compression_identifier = { ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned char )),   (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) +  compression_data->r_samplesInterval) *sizeof( unsigned char )};
        cl::Buffer compression_identifier_section = compression_data->compresion_identifier_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compression_identifier);
        
        queues[step % NUMBER_STREAMS][2].enqueueReadBuffer(compression_identifier_section,CL_TRUE,0,sizeof( unsigned char ) * compression_data->r_samplesInterval, compression_data->compresion_identifier_best_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_3);


        cl_buffer_region compression_identifier_internal = { ((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval * sizeof( unsigned char )),   (((step % NUMBER_STREAMS)  * compression_data->r_samplesInterval ) +  compression_data->r_samplesInterval) * sizeof( unsigned char )};
        cl::Buffer compression_identifier_internal_section = compression_data->compresion_identifier_internal_best->createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION, &compression_identifier_internal);
        
        queues[step % NUMBER_STREAMS][3].enqueueReadBuffer(compression_identifier_internal_section,CL_TRUE,0,sizeof( unsigned char ) * compression_data->r_samplesInterval, compression_data->compresion_identifier_best_internal_cpu + (compression_data->r_samplesInterval * step),NULL, t->t_device_host_4);

            
    }
    

    // sync GPU
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        for (unsigned int j = 0; j < MAXSIZE_NBITS; ++j) {
            queues[i][j].finish();
        }
         
    }
    //compression_data->queue->finish();
    
    // copy the data back and write to final data
    unsigned int compression_technique_identifier_size = 1;

    // define the size of the compression technique identifier base of n_bits size
    if (compression_data->n_bits < 3){
        compression_technique_identifier_size = 1;
    }
    else if (compression_data->n_bits < 5)
    {
        compression_technique_identifier_size = 2;
    }
    else if (compression_data->n_bits <= 8)
    {
        compression_technique_identifier_size = 3;
    }
    else if (compression_data->n_bits <= 16)
    {
        compression_technique_identifier_size = 4;
    }
    else 
    {
        compression_technique_identifier_size = 5;
    }

    // go for all of the steps
    for(int step = 0; step < compression_data->steps; ++step)
    {    
        if(compression_data->debug_mode){printf("Step %d\n",step);}
        // go per each block
        for(unsigned int block = 0; block < compression_data->r_samplesInterval; ++block)
        {  
            if(compression_data->debug_mode){printf("Block %d\n",block);}
            unsigned int final_compression_technique_identifier_size = compression_technique_identifier_size;
            // header
            const unsigned char CompressionTechniqueIdentifier = compression_data->compresion_identifier_best_cpu[block + (step * compression_data->r_samplesInterval)];
            if( compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)] == ZERO_BLOCK_ID ||  compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)] == SECOND_EXTENSION_ID)
            {
                // print CompressionTechniqueIdentifier
                final_compression_technique_identifier_size = compression_technique_identifier_size + 1;
            }
            writeWordChar(compression_data->OutputDataBlock, CompressionTechniqueIdentifier, final_compression_technique_identifier_size);
            
            // block compression
            const unsigned char best_compression_technique_identifier = compression_data->compresion_identifier_best_internal_cpu[block + (step * compression_data->r_samplesInterval)];
            unsigned int *data_pointer = compression_data->data_in_blocks_best_cpu + (block * compression_data->j_blocksize + (step * compression_data->r_samplesInterval * compression_data->j_blocksize));
            const unsigned int size = compression_data->size_block_best_cpu[block + (step *  compression_data->r_samplesInterval)];


            // print data_pointer data
            if(compression_data->debug_mode){
                printf("CompressionTechniqueIdentifier and Size :%u, %u\n",final_compression_technique_identifier_size, CompressionTechniqueIdentifier);
                if (best_compression_technique_identifier == ZERO_BLOCK_ID)
                {
                    for (int i = 0; i < compression_data->j_blocksize; ++i)
                    {
                        printf("%d ", 0);
                    }
                    printf("\n");
                }
                else
                {
                    if (best_compression_technique_identifier == SECOND_EXTENSION_ID)
                    {
                        for (int i = 0; i < compression_data->j_blocksize/2; ++i)
                        {
                            printf("%d ", data_pointer[i]);
                        }
                        printf("\n");
                    }
                    else
                    {
                        for (int i = 0; i < compression_data->j_blocksize; ++i)
                        {
                            printf("%d ", data_pointer[i]);
                        }
                        printf("\n");   
                        
                    }
                    
                }
                
           
                
            }


            if (best_compression_technique_identifier == ZERO_BLOCK_ID)
            {
                if(compression_data->debug_mode){printf("Zero block with size %d\n",size);}
                ZeroBlockWriter(compression_data->OutputDataBlock, size);
            }
            else if(best_compression_technique_identifier == NO_COMPRESSION_ID)
            {
                if(compression_data->debug_mode){printf("No compression with size %d\n",size);}
                NoCompressionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, compression_data->n_bits,data_pointer);
            }
            else if(best_compression_technique_identifier == FUNDAMENTAL_SEQUENCE_ID)
            {
                if(compression_data->debug_mode){printf("Fundamental sequence with size %d\n",size);}
                FundamentalSequenceWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, data_pointer);
            }
            else if(best_compression_technique_identifier == SECOND_EXTENSION_ID)
            {
                if(compression_data->debug_mode){printf("Second extension with size %d\n",size);}
                SecondExtensionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize/2,data_pointer);
            }
            else if(best_compression_technique_identifier >= SAMPLE_SPLITTING_ID)
            {
                if(compression_data->debug_mode){printf("Sample splitting with K %d and size %d\n",best_compression_technique_identifier - SAMPLE_SPLITTING_ID, size);}
                SampleSplittingWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, best_compression_technique_identifier - SAMPLE_SPLITTING_ID, data_pointer);
            }
            else
            {
                printf("Error: Unknown compression technique identifier\n");
            }
        }
        
       
      
    }

    T_STOP(t->t_test);


}



void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
//EMPTY In dis case not use because the copy happens asynchronous
}


void get_elapsed_time(
	compression_data_t *compression_data, 
	compression_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{
	    //cudaEventSynchronize(*compression_data->stop_memory_copy_host);
        float milliseconds_h_d = 0, milliseconds_d_h = 0;
        // memory transfer time host-device
        milliseconds_h_d = t->t_host_device->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_host_device->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        // kernel time 1
        long unsigned int application_miliseconds = (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
        //  memory transfer time device-host
        milliseconds_d_h = t->t_device_host_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        milliseconds_d_h += t->t_device_host_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        milliseconds_d_h += t->t_device_host_3->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host_3->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        milliseconds_d_h += t->t_device_host_4->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host_4->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        
        if (csv_format)
        {
            printf("%.10f;%lu;%.10f;\n", milliseconds_h_d,application_miliseconds,milliseconds_d_h);
        }
        else if (database_format)
        {
            printf("%.10f;%lu;%.10f;%ld;\n", milliseconds_h_d,application_miliseconds,milliseconds_d_h, timestamp);
        }
        else if(verbose_print)
        {
            printf("Elapsed time Host->Device: %.10f milliseconds\n", (float) milliseconds_h_d);
            printf("Elapsed time kernel: %lu milliseconds\n", application_miliseconds );
            printf("Elapsed time Device->Host: %.10f milliseconds\n", (float) milliseconds_d_h);
        }

}


void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
    //free(compression_data->InputDataBlock);
    //free(compression_data->OutputDataBlock);
    //TODO FREE rest of data
    //free(compression_data);
    // Free OpenCL memory
    delete compression_data->input_data;
    delete compression_data->output_data;
    delete compression_data->input_data_post_process;
    delete compression_data->missing_value;
    delete compression_data->missing_value_inverse;
    delete compression_data->zero_block_list;
    delete compression_data->zero_block_list_inverse;
    delete compression_data->compresion_identifier;
    delete compression_data->compresion_identifier_internal;
    delete compression_data->zero_block_list_status;
    delete compression_data->size_block;
    delete compression_data->data_in_blocks;
    delete compression_data->compresion_identifier_best;
    delete compression_data->compresion_identifier_internal_best;
    delete compression_data->size_block_best;
    delete compression_data->bit_block_best;
    delete compression_data->data_in_blocks_best;
    delete compression_data->context;
    delete compression_data->queue;
    delete compression_data->program;
    delete compression_data->default_device;



    // free CPU part
    free(compression_data->compresion_identifier_best_cpu);
    free(compression_data->compresion_identifier_best_internal_cpu);
    free(compression_data->size_block_best_cpu);
    free(compression_data->data_in_blocks_best_cpu);
    free(compression_data);





    
}