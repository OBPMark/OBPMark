/**
 * \file device.c
 * \brief Benchmark #122 OpenCL version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */

#include "device.h"
#include "processing.h"
#include "GEN_kernels.hcl"

cl::CommandQueue queues[NUMBER_STREAMS];

void syncstreams();

void init(
	compression_image_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	)
{
    init(compression_data,t, 0,0, device_name);
}



void init(
	compression_image_data_t *device_object,
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
    t->evt_copy_mains = new cl::Event; 
    t->evt_copy_auxiliar_float_1 = new cl::Event;
    t->evt_copy_auxiliar_float_2 = new cl::Event;
    t->evt = new cl::Event;
    t->evt_copy_back = new cl::Event;

    cl::Program::Sources sources;
    // load kernel from file
    std::string defines = "#define HIGHPASSFILTERSIZE " + std::to_string(HIGHPASSFILTERSIZE) +  "\n#define LOWPASSFILTERSIZE " + std::to_string(LOWPASSFILTERSIZE) + "\n#define BLOCKSIZEIMAGE " + std::to_string(BLOCKSIZEIMAGE) +"\n";
    kernel_code = defines + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    // build
    device_object->program = new cl::Program(*device_object->context,sources);
    if(device_object->program->build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<device_object->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }


}


bool device_memory_init(
	compression_image_data_t *compression_data
	)
{


    // Allocate the device image input
    compression_data->input_image_gpu = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));

    // input float image
    if (compression_data->type_of_compression)
    {
        // float opeation need to init the mid operation image
        compression_data->input_image_float = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,sizeof(float) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
        // mid procesing float operation
        compression_data->transformed_float = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,sizeof(float) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
        // Allocate the device low_filter
        compression_data->low_filter = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,LOWPASSFILTERSIZE * sizeof(float));
        // Allocate the device high_filter
        compression_data->high_filter = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,HIGHPASSFILTERSIZE * sizeof(float));
    }
    
    // output_image
    compression_data->transformed_image = new cl::Buffer(*compression_data->context,CL_MEM_READ_WRITE ,sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
    compression_data->output_image = (int *)malloc(sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows));
    
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        queues[i] = cl::CommandQueue(*compression_data->context,compression_data->default_device,NULL);
    }
    
    return true;
}


void copy_memory_to_device(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{

    queues[0].enqueueWriteBuffer(*compression_data->input_image_gpu,CL_TRUE,0,sizeof(int) * (compression_data->h_size + compression_data->pad_columns ) * (compression_data->w_size + compression_data->pad_rows), compression_data->input_image, NULL, t->evt_copy_mains);
    // float operation
    if (compression_data->type_of_compression)
    {
        queues[0].enqueueWriteBuffer(*compression_data->low_filter,CL_TRUE,0,sizeof(float) * LOWPASSFILTERSIZE, lowpass_filter_cpu, NULL, t->evt_copy_auxiliar_float_1);
        queues[0].enqueueWriteBuffer(*compression_data->high_filter,CL_TRUE,0,sizeof(float) * HIGHPASSFILTERSIZE, highpass_filter_cpu, NULL, t->evt_copy_auxiliar_float_2);
	}
    syncstreams();

}


void ccsds_wavelet_transform_2D(compression_image_data_t *device_object, unsigned int level)
{
     unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;

    cl::NDRange local;
    cl::NDRange global;
    
    unsigned int h_size_level = device_object->h_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_w_lateral = ((device_object->w_size + device_object->pad_rows) / (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ
    if (size_w_lateral <= BLOCK_SIZE*BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(size_w_lateral);
    }
    else
    {
        local = cl::NDRange(x_local * y_local);
        global = cl::NDRange(size_w_lateral);
    }
    for(unsigned int i = 0; i < h_size_level; ++i)
    {
        if(device_object->type_of_compression)
        {
            
            // float
            cl::Kernel kernel_wavelet_float=cl::Kernel(*device_object->program,"wavelet_transform_float");
            kernel_wavelet_float.setArg(0,*device_object->input_image_float);
            kernel_wavelet_float.setArg(1,*device_object->transformed_float);
            kernel_wavelet_float.setArg(2,size_w_lateral);
            kernel_wavelet_float.setArg(3,*device_object->low_filter);
            kernel_wavelet_float.setArg(4,*device_object->high_filter);
            kernel_wavelet_float.setArg(5,1);
            kernel_wavelet_float.setArg(6,((device_object->w_size + device_object->pad_rows) * i));

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_float,cl::NullRange,global,local, NULL, NULL);
            

        }
        else
        {

            // integer
            cl::Kernel kernel_wavelet_integer=cl::Kernel(*device_object->program,"wavelet_transform_int");
            kernel_wavelet_integer.setArg(0,*device_object->input_image_gpu);
            kernel_wavelet_integer.setArg(1,*device_object->transformed_image);
            kernel_wavelet_integer.setArg(2,size_w_lateral);
            kernel_wavelet_integer.setArg(3,1);
            kernel_wavelet_integer.setArg(4,((device_object->w_size + device_object->pad_rows) * i));

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer,cl::NullRange,global,local, NULL, NULL);


             cl::Kernel kernel_wavelet_integer_low=cl::Kernel(*device_object->program,"wavelet_transform_low_int");
            kernel_wavelet_integer_low.setArg(0,*device_object->input_image_gpu);
            kernel_wavelet_integer_low.setArg(1,*device_object->transformed_image);
            kernel_wavelet_integer_low.setArg(2,size_w_lateral);
            kernel_wavelet_integer_low.setArg(3,1);
            kernel_wavelet_integer_low.setArg(4,((device_object->w_size + device_object->pad_rows) * i));

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer_low,cl::NullRange,global,local, NULL, NULL);
        }
        
    }
    // SYSC all threads
    syncstreams();

    // encode columns
    unsigned int w_size_level = device_object->w_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_h_lateral = ((device_object->h_size + device_object->pad_columns)/ (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ

    if (size_h_lateral <= BLOCK_SIZE * BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(size_h_lateral);
    }
    else
    {
        local = cl::NDRange(x_local * y_local);
        global = cl::NDRange(size_h_lateral);
    }
    for(unsigned int i = 0; i < w_size_level; ++i)
    {
        
        if(device_object->type_of_compression)
        {
            cl::Kernel kernel_wavelet_float=cl::Kernel(*device_object->program,"wavelet_transform_float");
            kernel_wavelet_float.setArg(0,*device_object->transformed_float);
            kernel_wavelet_float.setArg(1,*device_object->input_image_float);
            kernel_wavelet_float.setArg(2,size_h_lateral);
            kernel_wavelet_float.setArg(3,*device_object->low_filter);
            kernel_wavelet_float.setArg(4,*device_object->high_filter);
            kernel_wavelet_float.setArg(5,device_object->w_size + device_object->pad_rows);
            kernel_wavelet_float.setArg(6,i);

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_float,cl::NullRange,global,local, NULL, NULL);

        
        }
        else
        {

            
            // integer
            cl::Kernel kernel_wavelet_integer=cl::Kernel(*device_object->program,"wavelet_transform_int");
            kernel_wavelet_integer.setArg(0,*device_object->transformed_image);
            kernel_wavelet_integer.setArg(1,*device_object->input_image_gpu);
            kernel_wavelet_integer.setArg(2,size_h_lateral);
            kernel_wavelet_integer.setArg(3,device_object->w_size + device_object->pad_rows);
            kernel_wavelet_integer.setArg(4,i);

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer,cl::NullRange,global,local, NULL, NULL);

            cl::Kernel kernel_wavelet_integer_low=cl::Kernel(*device_object->program,"wavelet_transform_low_int");
            kernel_wavelet_integer_low.setArg(0,*device_object->transformed_image);
            kernel_wavelet_integer_low.setArg(1,*device_object->input_image_gpu);
            kernel_wavelet_integer_low.setArg(2,size_h_lateral);
            kernel_wavelet_integer_low.setArg(3,device_object->w_size + device_object->pad_rows);
            kernel_wavelet_integer_low.setArg(4,i);

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer_low,cl::NullRange,global,local, NULL, NULL);
        }
    }
    // SYSC all threads
    syncstreams();
}


void process_benchmark(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{

	unsigned int h_size_padded = 0;
    unsigned int w_size_padded = 0;

    cl::NDRange local;
    cl::NDRange global;

    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;

	// create te new size
    h_size_padded = compression_data->h_size + compression_data->pad_rows;
    w_size_padded = compression_data->w_size + compression_data->pad_columns;
    unsigned int size = h_size_padded * w_size_padded;
    int  **transformed_image = NULL;
	transformed_image = (int**)calloc(h_size_padded, sizeof(int *));
	for(unsigned i = 0; i < h_size_padded; i++){
		transformed_image[i] = (int *)calloc(w_size_padded, sizeof(int));
	}

	unsigned int total_blocks =  (h_size_padded / BLOCKSIZEIMAGE )*(w_size_padded/ BLOCKSIZEIMAGE);
	int **block_string = NULL;
	block_string = (int **)calloc(total_blocks,sizeof(long *));
	for(unsigned int i = 0; i < total_blocks ; i++)
	{
		block_string[i] = (int *)calloc(BLOCKSIZEIMAGE * BLOCKSIZEIMAGE,sizeof(long));
	}
   
    // start computing the time
    T_START(t->t_test);
    T_START(t->t_dwt);
    // divide the flow depending of the type
    if(compression_data->type_of_compression)
    {
       // conversion to float
        if (size <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(size);
        }
        else
        {
            local = cl::NDRange(x_local * y_local);
            global = cl::NDRange(size);
        }

        cl::Kernel kernel_transform=cl::Kernel(*compression_data->program,"transform_image_to_float");
        kernel_transform.setArg(0,*compression_data->input_image_gpu);
        kernel_transform.setArg(1,*compression_data->input_image_float);
        kernel_transform.setArg(2,size);

        queues[0].enqueueNDRangeKernel(kernel_transform,cl::NullRange,global,local, NULL, NULL);
        syncstreams();
    }
    // launch WAVELET 2D
    unsigned int iteration = 0;
    while(iteration != (LEVELS_DWT )){ 
        ccsds_wavelet_transform_2D(compression_data, iteration);
        ++iteration;
    }
    // finish the conversion 
    if(compression_data->type_of_compression)
    {
        // float opperation
        // transformation
        // float opperation
        // transformation
        if (size <= BLOCK_SIZE)
        {
            local = cl::NullRange;
            global = cl::NDRange(size);
        }
        else
        {
            local = cl::NDRange(x_local * y_local);
            global = cl::NDRange(size);
        }

        cl::Kernel kernel_transform_int=cl::Kernel(*compression_data->program,"transform_image_to_int");
        kernel_transform_int.setArg(0,*compression_data->input_image_float);
        kernel_transform_int.setArg(1,*compression_data->input_image_gpu);
        kernel_transform_int.setArg(2,size);

        queues[0].enqueueNDRangeKernel(kernel_transform_int,cl::NullRange,global,local, NULL, NULL);
        //syncstreams();
    }
    
    // copy the memory to device to compute the compression
    queues[0].enqueueReadBuffer(*compression_data->input_image_gpu,CL_TRUE,0,sizeof(int) * size,compression_data->output_image, NULL, t->evt_copy_back);
    syncstreams();
    T_STOP(t->t_dwt);
    T_START(t->t_bpe);

    

	// read the image data
	for (unsigned int i = 0; i < h_size_padded; ++ i)
	{
		for (unsigned int j = 0; j < w_size_padded; ++j)
		{
			transformed_image[i][j] = compression_data->output_image [i * h_size_padded + j];
		}
	}

    if (!compression_data->type_of_compression)
	{
        coefficient_scaling(transformed_image, h_size_padded, w_size_padded);
	}
	coeff_regroup(transformed_image, h_size_padded, w_size_padded);

    // create block string
    build_block_string(transformed_image, h_size_padded, w_size_padded,block_string);
	
	// calculate the number of segments that are generated by the block_string if have some left from the division we add 1 to the number of segments
	unsigned int num_segments = 0;
	if (total_blocks % compression_data->segment_size != 0)
	{
		num_segments = (total_blocks / compression_data->segment_size) + 1;
	}
	else
	{
		num_segments = total_blocks / compression_data->segment_size;
	}
	// create the array of SegmentBitStream
	// compute the BPE
	compression_data->segment_list = (SegmentBitStream *)malloc(sizeof(SegmentBitStream) * num_segments);
	// init the array of SegmentBitStream
	for (unsigned int i = 0; i < num_segments; ++i)
	{
		compression_data->segment_list[i].segment_id = 0;
		compression_data->segment_list[i].num_total_bytes = 0;
		compression_data->segment_list[i].first_byte = NULL;
		compression_data->segment_list[i].last_byte = NULL;
	}
	compression_data->number_of_segments = num_segments;
	compute_bpe(compression_data, block_string, num_segments);


	T_STOP(t->t_bpe);
	T_STOP(t->t_test);
	
	// clean image
	for(unsigned int i = 0; i < h_size_padded; i++){
			free(transformed_image[i]);
		}
	free(transformed_image);
	// free block_string
	for(unsigned int i = 0; i < total_blocks; i++){
			free(block_string[i]);
		}
	free(block_string);
}


void copy_memory_to_host(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	// empty
}


void get_elapsed_time(
	compression_image_data_t *compression_data, 
	compression_time_t *t,
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{

    float milliseconds_h_d = 0, milliseconds_d_h = 0;
    milliseconds_h_d = t->evt_copy_mains->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->evt_copy_mains->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    milliseconds_h_d += t->evt_copy_auxiliar_float_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->evt_copy_auxiliar_float_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    milliseconds_h_d += t->evt_copy_auxiliar_float_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->evt_copy_auxiliar_float_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    milliseconds_d_h = t->evt_copy_back->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->evt_copy_back->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    

	if (csv_format)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;\n", (float) milliseconds_h_d, elapsed_time, (float) milliseconds_h_d);
	}
	else if (database_format)
	{
		
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;%ld;\n", (float) milliseconds_h_d, elapsed_time, (float) milliseconds_h_d, timestamp);
	}
	else if(verbose_print)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("Elapsed time Host->Device: %.10f ms\n", (float) milliseconds_h_d);
		printf("Elapsed time kernel: %.10f ms\n", elapsed_time );
		printf("Elapsed time Device->Host: %.10f ms\n", (float) milliseconds_h_d);
	}
    
}



void clean(
	compression_image_data_t *compression_data,
	compression_time_t *t
	)
{
	clean_segment_bit_stream(compression_data->segment_list, compression_data->number_of_segments);
	// clean the auxiliary structures
	free(compression_data->input_image);
	free(compression_data->segment_list);
    free(compression_data->output_image);
	free(compression_data);
}

void syncstreams(){
     for (unsigned int x = 0; x < NUMBER_STREAMS; ++x) {queues[x].finish();}
}