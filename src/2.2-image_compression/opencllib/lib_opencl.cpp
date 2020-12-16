
#include "../lib_functions.h"
#include "GEN_kernel.hcl"



cl::CommandQueue queues[NUMBER_STREAMS];

int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns);
void ccsds_wavelet_transform_2D(DataObject *device_object, unsigned int level);
void header_inilization(header_struct *header_pointer, BOOL DWT_type, unsigned int pad_rows, unsigned int image_width, BOOL first, BOOL last);
void encode_dc(block_struct* list_of_process_data, long *block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks,unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer);
void dpcm_dc_mapper(unsigned long *shifted_dc, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_entropy_encoder(block_struct* list_of_process_data, unsigned int block_position, unsigned short *dc_remainer, unsigned char quantization_factor, header_struct *header_pointer, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N);
void dc_encoder(unsigned int gaggle_id, unsigned int block_position, block_struct* list_of_process_data, header_struct *header_pointer, unsigned long *dc_mapper, short N, int star_index, int gaggles, int max_k, int id_length);
void convert_to_bits_segment(FILE *ofp, block_struct segment_data);
unsigned long conversion_twos_complement(long original, short leftmost);


void syncstreams();


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
    device_object->evt_copy_mains = new cl::Event; 
    device_object->evt_copy_auxiliar_float_1 = new cl::Event;
    device_object->evt_copy_auxiliar_float_2 = new cl::Event;
    device_object->evt = new cl::Event;
    device_object->evt_copy_back = new cl::Event;

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

bool device_memory_init(DataObject *device_object){

    // Allocate the device image input
    device_object->input_image = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));

    // input float image
    if (device_object->type)
    {
        // float opeation need to init the mid operation image
        device_object->input_image_float = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(float) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
        // mid procesing float operation
        device_object->transformed_float = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(float) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
        // Allocate the device low_filter
        device_object->low_filter = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,LOWPASSFILTERSIZE * sizeof(float));
        // Allocate the device high_filter
        device_object->high_filter = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,HIGHPASSFILTERSIZE * sizeof(float));
    }
    device_object->output_image = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    
    // output_image
    device_object->transformed_image = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    device_object->final_transformed_image = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));
    device_object->coeff_image_regroup = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows));

	const unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
    const unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
    
    device_object->block_string = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(long) * block_string_size);
    
    
    for (unsigned int i = 0; i < NUMBER_STREAMS; ++i) {
        queues[i] = cl::CommandQueue(*device_object->context,device_object->default_device,CL_QUEUE_PROFILING_ENABLE);
    }
    
    return true;


}

void copy_data_to_gpu(DataObject *device_object, int* image_data)
{
    
    queues[0].enqueueWriteBuffer(*device_object->input_image,CL_TRUE,0,sizeof(int) * (device_object->h_size + device_object->pad_columns ) * (device_object->w_size + device_object->pad_rows), image_data, NULL, device_object->evt_copy_mains);
    // float operation
    if (device_object->type)
    {
        queues[0].enqueueWriteBuffer(*device_object->low_filter,CL_TRUE,0,sizeof(float) * LOWPASSFILTERSIZE, lowpass_filter_cpu, NULL, device_object->evt_copy_auxiliar_float_1);
        queues[0].enqueueWriteBuffer(*device_object->high_filter,CL_TRUE,0,sizeof(float) * HIGHPASSFILTERSIZE, highpass_filter_cpu, NULL, device_object->evt_copy_auxiliar_float_2);
	}
	
}

void copy_data_to_cpu(DataObject *device_object, long* block_string)
{   
	unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
	unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
	
    queues[0].enqueueReadBuffer(*device_object->block_string,CL_TRUE,0,block_string_size * sizeof(long),block_string, NULL, device_object->evt_copy_back);
    
   
}


void encode_engine(DataObject *device_object, int* image_data)
{
    unsigned int h_size_padded = 0;
    unsigned int w_size_padded = 0;

    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;

    cl::NDRange local;
    cl::NDRange global;

	// create te new size
    h_size_padded = device_object->h_size + device_object->pad_rows;
    w_size_padded = device_object->w_size + device_object->pad_columns;
    unsigned int size = h_size_padded * w_size_padded;
    //int* image_data = readBMP(device_object->filename_input, device_object->pad_rows, device_object->pad_columns);
    // start computing the time
    copy_data_to_gpu(device_object, image_data);
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_dwt);
    if(device_object->type)
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

        cl::Kernel kernel_transform=cl::Kernel(*device_object->program,"transform_image_to_float");
        kernel_transform.setArg(0,*device_object->input_image);
        kernel_transform.setArg(1,*device_object->input_image_float);
        kernel_transform.setArg(2,size);

        queues[0].enqueueNDRangeKernel(kernel_transform,cl::NullRange,global,local, NULL, NULL);
        syncstreams();
    }
    // launch WAVELET 2D
    unsigned int iteration = 0;
    while(iteration != (LEVELS_DWT )){ 
        ccsds_wavelet_transform_2D(device_object, iteration);
        ++iteration;
    }
    // finish the conversion 
    if(device_object->type)
    {
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

        cl::Kernel kernel_transform_int=cl::Kernel(*device_object->program,"transform_image_to_int");
        kernel_transform_int.setArg(0,*device_object->input_image_float);
        kernel_transform_int.setArg(1,*device_object->output_image);
        kernel_transform_int.setArg(2,size);

        queues[0].enqueueNDRangeKernel(kernel_transform_int,cl::NullRange,global,local, NULL, NULL);
        syncstreams();
    }
    else
    {
        // integer operation
        // copy the memory
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

        cl::Kernel kernel_copy_int=cl::Kernel(*device_object->program,"copy_image_to_int");
        kernel_copy_int.setArg(0,*device_object->input_image);
        kernel_copy_int.setArg(1,*device_object->output_image);
        kernel_copy_int.setArg(2,size);

        queues[0].enqueueNDRangeKernel(kernel_copy_int,cl::NullRange,global,local, NULL, NULL);
        syncstreams();
    }
	syncstreams();
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_dwt);
    // FINSH DWT 2D
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->start_bpe);
    // coeff_regroup
    if (h_size_padded/8 <= BLOCK_SIZE && w_size_padded/8 <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(h_size_padded/8, w_size_padded/8);
    }
    else
    {
        local = cl::NDRange(x_local , y_local);
        global = cl::NDRange(h_size_padded/8, w_size_padded/8);
    }
    cl::Kernel kernel_coaeff=cl::Kernel(*device_object->program,"coeff_regroup");
    kernel_coaeff.setArg(0,*device_object->output_image);
    kernel_coaeff.setArg(1,*device_object->coeff_image_regroup);
    kernel_coaeff.setArg(2,h_size_padded);
    kernel_coaeff.setArg(3,w_size_padded);

    queues[0].enqueueNDRangeKernel(kernel_coaeff,cl::NullRange,global,local, NULL, NULL);

    // block string creation
    unsigned int block_h = h_size_padded / BLOCKSIZEIMAGE;
	unsigned int block_w = w_size_padded / BLOCKSIZEIMAGE;

    if (block_h <= BLOCK_SIZE && block_w <= BLOCK_SIZE)
    {
        local = cl::NullRange;
        global = cl::NDRange(block_h, block_w);
    }
    else
    {
        local = cl::NDRange(x_local , y_local);
        global = cl::NDRange(block_h, block_w);
    }
    cl::Kernel kernel_block_string=cl::Kernel(*device_object->program,"block_string_creation");
    kernel_coaeff.setArg(0,*device_object->coeff_image_regroup);
    kernel_coaeff.setArg(1,*device_object->block_string);
    kernel_coaeff.setArg(2,h_size_padded);
    kernel_coaeff.setArg(3,w_size_padded);

    queues[0].enqueueNDRangeKernel(kernel_coaeff,cl::NullRange,global,local, NULL, NULL);
    syncstreams();
    // accelerated GPU procesing finish
    unsigned int total_blocks =  ((device_object->h_size + device_object->pad_columns ) / BLOCKSIZEIMAGE )*((device_object->w_size + device_object->pad_rows)/ BLOCKSIZEIMAGE);
    unsigned int block_string_size =  total_blocks * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE;
    
	long *block_string = (long *)malloc(sizeof(long) * block_string_size);
    copy_data_to_cpu(device_object, block_string);
    // loop over all blocks in Segment size portions
	BOOL DWT_type = device_object->type ? TRUE : FALSE;
	// create the list of data to print
	unsigned int segment = 0;
	unsigned int number_of_segment = total_blocks / SEGMENTSIZE;
	block_struct *list_of_process_data = NULL;
	list_of_process_data = (block_struct *) calloc(number_of_segment, sizeof(block_struct)); 
	for (unsigned int block_counter = 0; block_counter < total_blocks; block_counter += SEGMENTSIZE)
	{
		// Step 3 create the header data
		// create the header
        header_struct *ptr_header = (header_struct *)malloc(sizeof(header_struct));
		// inicialize header
		
		if (block_counter == 0){
			// first block
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, TRUE, FALSE);
		}
		else if (block_counter + SEGMENTSIZE == total_blocks)
		{
			// last block
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, TRUE);
		}
		else if (block_counter + SEGMENTSIZE > total_blocks)
		{			
			// Special sitiuation when the number of blocks per segment are not the same
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, TRUE);
			ptr_header->header.part1.part_3_flag = TRUE;
			ptr_header->header.part3.seg_size_blocks_20bits = total_blocks - block_counter;

			ptr_header->header.part1.part_2_flag = TRUE;
			ptr_header->header.part2.seg_byte_limit_27bits = BITSPERFIXEL * (total_blocks - block_counter) * 64/8;
		}
		else
		{
			header_inilization(ptr_header, DWT_type, device_object->pad_rows, device_object->w_size, FALSE, FALSE);
		}
		// Step 4 encode DC
		// array that stores the maximun bit in AC in each of the blocks in a segment
		unsigned short *bit_max_in_segment = NULL;
		// get the numer of blocks in this segment
		unsigned int blocks_in_segment = block_counter + SEGMENTSIZE > total_blocks ? total_blocks - block_counter : SEGMENTSIZE ;
		bit_max_in_segment = (unsigned short*)calloc(blocks_in_segment, sizeof(unsigned short));
		unsigned char quantization_factor = 0;
		unsigned short *dc_remainer = NULL;
        dc_remainer = (unsigned short*)calloc(blocks_in_segment, sizeof(unsigned short));
		encode_dc(list_of_process_data, block_string, ptr_header, block_counter, total_blocks,bit_max_in_segment, &quantization_factor,dc_remainer);
		// copy data
		list_of_process_data[segment].dc_remainer = dc_remainer;
		list_of_process_data[segment].quantization_factor = quantization_factor;
		list_of_process_data[segment].header = (header_struct_base *) calloc( 1,sizeof(header_struct_base));
		list_of_process_data[segment].header->part1 = ptr_header->header.part1;
		list_of_process_data[segment].header->part2 = ptr_header->header.part2;
		list_of_process_data[segment].header->part3 = ptr_header->header.part3;
		list_of_process_data[segment].header->part4 = ptr_header->header.part4;
		++segment;
		//free(ptr_header);

	}
	// end processing
    clock_gettime(CLOCK_MONOTONIC_RAW, &device_object->end_bpe);
    // store the data
	FILE *ofp;
    ofp = fopen(device_object->filename_output, "w");

    if (ofp == NULL) {
    fprintf(stderr, "Can't open output file %s!\n", device_object->filename_output);
    exit(1);
    }
	// loop over the data
	for (unsigned int i = 0; i < number_of_segment; ++i)
	{
		convert_to_bits_segment(ofp, list_of_process_data[i]);
	}

	fclose(ofp);

}


void ccsds_wavelet_transform_2D(DataObject *device_object, unsigned int level)
{
    unsigned int x_local= BLOCK_SIZE;
    unsigned int y_local= BLOCK_SIZE;

    cl::NDRange local;
    cl::NDRange global;
    
    unsigned int h_size_level = device_object->h_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_w_lateral = ((device_object->w_size + device_object->pad_rows) / (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ
    
    if (size_w_lateral <= BLOCK_SIZE)
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
        if(device_object->type)
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

            cl::Kernel kernel_wavelet_integer_low=cl::Kernel(*device_object->program,"wavelet_transform_low_int");
            kernel_wavelet_integer_low.setArg(0,*device_object->input_image);
            kernel_wavelet_integer_low.setArg(1,*device_object->transformed_image);
            kernel_wavelet_integer_low.setArg(2,size_w_lateral);
            kernel_wavelet_integer_low.setArg(3,1);
            kernel_wavelet_integer_low.setArg(4,((device_object->w_size + device_object->pad_rows) * i));

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer_low,cl::NullRange,global,local, NULL, NULL);
            // integer

            cl::Kernel kernel_wavelet_integer=cl::Kernel(*device_object->program,"wavelet_transform_int");
            kernel_wavelet_integer.setArg(0,*device_object->input_image);
            kernel_wavelet_integer.setArg(1,*device_object->transformed_image);
            kernel_wavelet_integer.setArg(2,size_w_lateral);
            kernel_wavelet_integer.setArg(3,1);
            kernel_wavelet_integer.setArg(4,((device_object->w_size + device_object->pad_rows) * i));

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer,cl::NullRange,global,local, NULL, NULL);
        }
        
    }
    // SYSC all threads
    syncstreams();
    // encode columns
    unsigned int w_size_level = device_object->w_size / (1<<level);  // power of two (づ｡◕‿‿◕｡)づ
    unsigned int size_h_lateral = ((device_object->h_size + device_object->pad_columns)/ (1 <<level))/2; // power of two (づ｡◕‿‿◕｡)づ

    if (size_h_lateral <= BLOCK_SIZE)
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
        
        if(device_object->type)
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

            cl::Kernel kernel_wavelet_integer_low=cl::Kernel(*device_object->program,"wavelet_transform_low_int");
            kernel_wavelet_integer_low.setArg(0,*device_object->transformed_image);
            kernel_wavelet_integer_low.setArg(1,*device_object->input_image);
            kernel_wavelet_integer_low.setArg(2,size_h_lateral);
            kernel_wavelet_integer_low.setArg(3,device_object->w_size + device_object->pad_rows);
            kernel_wavelet_integer_low.setArg(4,i);

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer_low,cl::NullRange,global,local, NULL, NULL);
            // integer

            cl::Kernel kernel_wavelet_integer=cl::Kernel(*device_object->program,"wavelet_transform_int");
            kernel_wavelet_integer.setArg(0,*device_object->input_image);
            kernel_wavelet_integer.setArg(1,*device_object->transformed_image);
            kernel_wavelet_integer.setArg(2,size_h_lateral);
            kernel_wavelet_integer.setArg(3,device_object->w_size + device_object->pad_rows);
            kernel_wavelet_integer.setArg(4,i);

            queues[i % NUMBER_STREAMS].enqueueNDRangeKernel(kernel_wavelet_integer,cl::NullRange,global,local, NULL, NULL);
        }
    }
    // SYSC all threads
    syncstreams();

}


/*int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns)
{
	BMP Image;
   	Image.ReadFromFile( filename );
    int size_output =  Image.TellWidth() * Image.TellHeight();
	int  *data_bw = NULL;
	data_bw = (int*)calloc((Image.TellHeight() + pad_rows ) * (Image.TellWidth() + pad_columns), sizeof(int));
   	// convert each pixel to greyscale
   	for( int i=0 ; i < Image.TellHeight() ; i++)
   	{
    	for( int j=0 ; j < Image.TellWidth() ; j++)
    	{
			data_bw[i * Image.TellWidth() + j] =  (Image(j,i)->Red + Image(j,i)->Green + Image(j,i)->Blue)/3;
    	}
   }
   //we need to duplicate rows and columns to be in BLOCKSIZEIMAGE
   for(unsigned int i = 0; i < pad_rows ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_columns; j++)
			data_bw[(i + Image.TellHeight()) * Image.TellWidth() + j] = data_bw[(Image.TellHeight() - 1)* Image.TellWidth() + j];
	}

	for(unsigned int i = 0; i < pad_columns ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_rows ; j++)
			data_bw[(j)* Image.TellWidth() + (i + Image.TellWidth())] = data_bw[j * Image.TellWidth() + ( Image.TellWidth() - 1)];
	}
    
   return data_bw;
}*/


void get_elapsed_time(DataObject *device_object, bool csv_format){
    device_object->evt_copy_back->wait();
    float elapsed_h_d = 0, elapsed_d_h = 0;
    unsigned long  milliseconds = 0, miliseconds_bpe = 0;
    elapsed_h_d =  device_object->evt_copy_mains->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copy_mains->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    if(device_object->type)
    {
        elapsed_h_d += device_object->evt_copy_auxiliar_float_1->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copy_auxiliar_float_1->getProfilingInfo<CL_PROFILING_COMMAND_START>();
        elapsed_h_d += device_object->evt_copy_auxiliar_float_2->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copy_auxiliar_float_2->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    }
    
    // TODO add the rest of the kernels
    milliseconds =  (device_object->end_dwt.tv_sec - device_object->start_dwt.tv_sec) * 1000 + (device_object->end_dwt.tv_nsec - device_object->start_dwt.tv_nsec) / 1000000;
	miliseconds_bpe = (device_object->end_bpe.tv_sec - device_object->start_bpe.tv_sec) * 1000 + (device_object->end_bpe.tv_nsec - device_object->start_bpe.tv_nsec) / 1000000;


    elapsed_d_h = device_object->evt_copy_back->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copy_back->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    


    if (csv_format){
         printf("%.10f;%lu;%lu;%.10f;\n", elapsed_h_d / 1000000.0,milliseconds, miliseconds_bpe,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel DWT: %lu miliseconds\n", milliseconds);
         printf("Elapsed time kernel BPE: %lu miliseconds\n", miliseconds_bpe);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }
}
//encoder part
void header_inilization(header_struct *header_pointer, BOOL DWT_type ,unsigned int pad_rows, unsigned int image_width, BOOL first, BOOL last)
{
	header_pointer->header.part1.start_img_flag = first;
	header_pointer->header.part1.end_img_flag = last;
	header_pointer->header.part1.segment_count_8bits = 0;
	header_pointer->header.part1.bit_depth_ac_5bits = 0;
	header_pointer->header.part1.bit_depth_dc_5bits = 0;
	header_pointer->header.part1.part_2_flag = TRUE;
	header_pointer->header.part1.part_3_flag = TRUE;
	header_pointer->header.part1.part_4_flag = TRUE;
	header_pointer->header.part1.pad_rows_3bits = pad_rows;
	header_pointer->header.part1.reserved_5bits = 0;

	header_pointer->header.part2.seg_byte_limit_27bits = 0; 
	header_pointer->header.part2.dc_stop = FALSE;
	header_pointer->header.part2.bit_plane_stop_5bits = 0;
	header_pointer->header.part2.stage_stop_2bits = 3; // 2 bits, transform input data quantization. TODO chech if should be 0
	header_pointer->header.part2.use_fill = header_pointer->header.part2.seg_byte_limit_27bits == 0 ? FALSE : TRUE;
	header_pointer->header.part2.reserved_4bits = 0;

	header_pointer->header.part3.seg_size_blocks_20bits = SEGMENTSIZE;
	header_pointer->header.part3.opt_dc_select = TRUE;
	header_pointer->header.part3.opt_ac_select = TRUE;
	header_pointer->header.part3.reserved_2bits = 0;

	header_pointer->header.part4.dwt_type = TRUE;
	header_pointer->header.part4.reserved_2bits = 0; 
	header_pointer->header.part4.signed_pixels = FALSE;
	header_pointer->header.part4.pixel_bit_depth_4bits = 8;
	header_pointer->header.part4.image_with_20bits = image_width;
	header_pointer->header.part4.transpose_img = FALSE;
	header_pointer->header.part4.code_word_length = 000;
	header_pointer->header.part4.custom_wt_flag = FALSE;
	header_pointer->header.part4.custom_wt_HH1_2bits = 0;
	header_pointer->header.part4.custom_wt_HL1_2bits = 0;
	header_pointer->header.part4.custom_wt_LH1_2bits = 0;
	header_pointer->header.part4.custom_wt_HH2_2bits = 0;
	header_pointer->header.part4.custom_wt_HL2_2bits = 0;
	header_pointer->header.part4.custom_wt_LH2_2bits = 0;
	header_pointer->header.part4.custom_wt_HH3_2bits = 0;
	header_pointer->header.part4.custom_wt_HL3_2bits = 0;
	header_pointer->header.part4.custom_wt_LH3_2bits = 0;
	header_pointer->header.part4.custom_wt_LL3_2bits = 0;
	header_pointer->header.part4.reserved_11bits = 0;
}

void encode_dc(block_struct* list_of_process_data, long *block_string, header_struct *header_pointer ,unsigned int block_position, unsigned int total_blocks, unsigned short *bit_max_in_segment, unsigned char *quantization_factor, unsigned short *dc_remainer)
{
	long  max_dc_per_segment = 0;

	signed long dc_min = 0x10000;
	signed long dc_max = -0x10000;

	signed long max_ac_segment = 0;

	// Start of the calculations to get the bit depth in the segment in the DC range and AC
	unsigned int blocks_in_segment = block_position + SEGMENTSIZE > total_blocks ? total_blocks - block_position : SEGMENTSIZE ;
	// loop over a segment щ（ﾟДﾟщ）
	// determine max position
	unsigned int max_position = block_position + SEGMENTSIZE > total_blocks ? total_blocks : block_position + SEGMENTSIZE; // min operation (づ｡◕‿‿◕｡)づ
	// one poisition stores a segment so we loop over all of the DC values
	// this loop is for extract all of the maximun values of the DC operands and AC
	for (unsigned int block_position_in_segment = block_position; block_position_in_segment < max_position; ++block_position_in_segment)
	{
        if(TOPOSITIVE(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0]) > TOPOSITIVE(max_dc_per_segment))
		{
            // update the value of max dc in this segment
		    max_dc_per_segment = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0];
		}
		
		
		if(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0] > dc_max)
		{
			dc_max = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0];
		}

		if(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0] > dc_min)
		{
			dc_min = block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +0];
		}

		signed long ac_bit_max_one_block = 0; 
		for(unsigned int i = 1; i < BLOCKSIZEIMAGE * BLOCKSIZEIMAGE; ++i)
		{
			unsigned int abs_ac = 0;
			abs_ac = TOPOSITIVE(block_string[block_position_in_segment * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  +i]);
			if ((signed long)abs_ac > ac_bit_max_one_block)
			{
				ac_bit_max_one_block = abs_ac;
			}
			if ((signed long)abs_ac > max_ac_segment)
			{
				max_ac_segment = abs_ac;
			}
		}
		// extrat the numer of bit maximun per segment for high AC segment (╯°□°）╯︵ ┻━┻ 

		while (ac_bit_max_one_block > 0)
		{
			ac_bit_max_one_block >>= 1; // divide by two
			++bit_max_in_segment[block_position_in_segment - block_position];
		}
    }
	// finish the part of extracting data from all of the blocks in a segment
	header_pointer->header.part1.bit_depth_ac_5bits = 0;
	header_pointer->header.part1.bit_depth_dc_5bits = 0;
	// get the maximun bit depth in ac in all of the blocks
	while (max_ac_segment > 0)
	{
		max_ac_segment >>= 1;
		++header_pointer->header.part1.bit_depth_ac_5bits;
	}
	// get the maximuin DC depth in all blocks
	if (dc_min >= 0)
	{
		max_dc_per_segment = dc_max;
	}
	else if (dc_max <= 0)
	{
		max_dc_per_segment = dc_min;
	}
	else if (dc_max >= TOPOSITIVE(dc_min))
	{
		max_dc_per_segment = dc_max;
	}
	else
	{
		max_dc_per_segment = dc_min;
	}
	
	if (max_dc_per_segment >=0)
	{
		while (max_dc_per_segment > 0)
		{
			max_dc_per_segment >>=1; // divide by two (づ｡◕‿‿◕｡)づ
			++header_pointer->header.part1.bit_depth_dc_5bits;
		}
		
	}
	else
	{
		unsigned long temp = -max_dc_per_segment; // chaneg sing of DC in other to extra the numer of bits
		while (temp > 0)
		{
			temp >>=1; // divide by two (づ｡◕‿‿◕｡)づ
			++header_pointer->header.part1.bit_depth_dc_5bits;
		}
		if( (1 << (header_pointer->header.part1.bit_depth_dc_5bits -1)) == - max_dc_per_segment)
		{
			--header_pointer->header.part1.bit_depth_dc_5bits;
		}
	}
	// include the sign bit
	++header_pointer->header.part1.bit_depth_dc_5bits;

	// END of the calculations to get the bit depth in the segment in the DC range and AC
	// all this for get the bit_depth (╯°□°）╯︵ ┻━┻ 

	// copy the values to variables
	unsigned short bit_depth_dc = header_pointer->header.part1.bit_depth_dc_5bits;
	unsigned short bit_depth_ac = header_pointer->header.part1.bit_depth_ac_5bits;
	
	// convert the header to binary
	// TODO

	// start with the DC conversion ¯\\_(ツ)_/¯  I hoppe
	//////////////// Start of the table 4-8. Pages 44,45 CCSDS 122 ///////////////
	unsigned short quantization_of_factor_q_prime= 0;
	if (bit_depth_dc <= 3)
	{
		quantization_of_factor_q_prime = 0;
	}
	else if (((bit_depth_dc - (1 + (bit_depth_ac >> 1))) <= 1 ) && (bit_depth_dc > 3))
	{
		quantization_of_factor_q_prime = bit_depth_dc - 3;
	}
	else if (((bit_depth_dc - (1 + (bit_depth_ac >> 1))) > 10 ) && (bit_depth_dc > 3))
	{
		quantization_of_factor_q_prime = bit_depth_dc - 10;
	}
	else
	{
		quantization_of_factor_q_prime = 1 + (bit_depth_ac>>1);
	}
	//////////////// End of the table 4-8. Pages 44,45 CCSDS 122 ///////////////
	// Point 4.3.1.3 
	*quantization_factor = quantization_of_factor_q_prime > header_pointer->header.part4.custom_wt_LL3_2bits ? quantization_of_factor_q_prime : header_pointer->header.part4.custom_wt_LL3_2bits; // max operation (づ｡◕‿‿◕｡)づ


	// Shift of the DC component
	// k consists of quantization_factor ones in the least significant bits ¯\\_(ツ)_/¯ i suposse
	
	unsigned int k = (1 << *quantization_factor)-1;
	
	// shift the DC component to the right
	unsigned long *shifted_dc = NULL;
	shifted_dc = (unsigned long*)calloc(blocks_in_segment, sizeof(unsigned long));
	
	for(unsigned int i = 0; i < blocks_in_segment; ++i)
	{
		// convbert the DC value to twos complement
		unsigned long new_value = conversion_twos_complement (block_string[i * BLOCKSIZEIMAGE * BLOCKSIZEIMAGE  + 0], bit_depth_dc);
		shifted_dc[i] = (new_value >> *quantization_factor);
		dc_remainer[i] = (unsigned short) (new_value & k); // Binary AND ¯\\_(ツ)_/¯ 
	}
	
	unsigned char N = header_pointer->header.part1.bit_depth_dc_5bits - *quantization_factor > 1 ? header_pointer->header.part1.bit_depth_dc_5bits - *quantization_factor: 1; // max operation (づ｡◕‿‿◕｡)づ
	//The DC coefficient quantization rules in table 4-8 limit the value of N to be
    //within 10, and thus coding options are defined only up to N =10.||  4.3.2.1

	//When N is 1, each quantized DC coefficient c'm consists of a single bit. In this case,
	//the coded quantized DC coefficients for a segment consist of these bits, concatenated
	//together. Otherwise N>1 and the quantized DC coefficients in a segment, c'm, shall be
	//encoded using the procedure described below.  || 4.3.2.2
	//The first quantized DC coefficient for every sequence of S consecutive coefficients,
	//referred to as a reference sample, shall be written to the encoded bitstream directly (i.e.,
	//without any further processing or encoding) || 4.3.2.3
	list_of_process_data[block_position/SEGMENTSIZE].N = N;
	list_of_process_data[block_position/SEGMENTSIZE].shifted_dc_1bit = shifted_dc;
	if (N == 1)
	{

	}
	else
	{
		// Sample-spit entropy for bit shifted DC's || 4.3.2.6
		unsigned long *dc_mapper = NULL;
		dc_mapper = (unsigned long*)calloc(blocks_in_segment, sizeof(unsigned long));
		dpcm_dc_mapper(shifted_dc,dc_mapper,blocks_in_segment,N);
	
		dc_entropy_encoder(list_of_process_data,block_position ,dc_remainer,*quantization_factor,header_pointer,dc_mapper,blocks_in_segment, N);
		list_of_process_data[block_position/SEGMENTSIZE].dc_mapper = dc_mapper;
	}

	
}

void dpcm_dc_mapper(unsigned long *shifted_dc, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N)
{
	long theta = 0;
	unsigned long bits1 = 0; 
	long x_min = 0;
	long x_max = 0;

	long * dc_diff = NULL; 
	dc_diff = (long *)calloc(blocks_in_segment,sizeof(long));
	// change the first position of the shifted_dc to mapp that is 0
	dc_mapper[0] = shifted_dc[0];
	for (unsigned int i = 0; i < (N -1); ++i)
	{
		bits1 = (bits1 <<1)+1;
	}

	if((shifted_dc[0] & (1 << (N -1))) > 0) // minus 0
	{
		shifted_dc[0] = -(short)( ((shifted_dc[0]^bits1) & bits1) + 1 ); // negative 
	}
	else
	{ 
		shifted_dc[0] =  shifted_dc[0];// positive. 
	}

	dc_diff[0] = shifted_dc[0];
	for (unsigned int i = 1; i < blocks_in_segment; ++i)
	{
		if((shifted_dc[i] & (1 << (N -1))) > 0) // minus 0
		{
			shifted_dc[i] = -(short)( ((shifted_dc[i]^bits1) & bits1) + 1 );
		}
		else
		{
			shifted_dc[0] =  shifted_dc[0];// positive.
		}
		// get the diff
		dc_diff[i] = shifted_dc[i] - shifted_dc[i - 1];
	}

	for (unsigned int i = 1; i <blocks_in_segment; ++i) // 4.3.2.4 equations 18 and 19
	{
		theta = shifted_dc[i - 1] - x_min > x_max - shifted_dc[i - 1] ? x_max - shifted_dc[i - 1] : shifted_dc[i - 1] - x_min; // min operation (づ｡◕‿‿◕｡)づ
		if(dc_diff[i] >= 0 && dc_diff[i] <= theta)
		{
			dc_mapper[i] = 2*dc_diff[i];
		}
		else if (dc_diff[i] < 0 && dc_diff[i] >= -theta)
		{
			dc_mapper[i] = 2 * abs(dc_diff[i]) -1;
		}
		else
		{
			dc_mapper[i] = theta + abs(dc_diff[i]);
		}
	
	}
	free(dc_diff);
}

void dc_entropy_encoder(block_struct* list_of_process_data, unsigned int block_position, unsigned short *dc_remainer, unsigned char quantization_factor, header_struct *header_pointer, unsigned long *dc_mapper, unsigned int blocks_in_segment, short N)
{
	int max_k = 0;
	int id_length = 0;


	// CODE Option identifiers Table 4-9 4.3.2.7
	if (N == 2)
	{
		max_k = 0;
		id_length = 1;
	}
	else if (N <= 4)
	{
		max_k = 2;
		id_length = 2;
	}
	else if ( N <= 8)
	{
		max_k = 6;
		id_length = 3;
	}
	else if (N <= 10)
	{
		max_k = 8;
		id_length = 4;
	}
	
	// Gaggle processing
	unsigned int gaggle_index = 0;
	unsigned int gaggles = 0;
	unsigned int total_gaggles = blocks_in_segment / GAGGLE_SIZE;
	unsigned int gaggle_id = 0;
	list_of_process_data[block_position/SEGMENTSIZE].bits_min_k = id_length;
	list_of_process_data[block_position/SEGMENTSIZE].min_k = (int *)calloc(total_gaggles,sizeof(int));

	while(gaggle_index < blocks_in_segment)
	{
		gaggles = GAGGLE_SIZE > (blocks_in_segment - gaggle_index) ? (blocks_in_segment - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
		dc_encoder(gaggle_id, block_position, list_of_process_data, header_pointer, dc_mapper, N, gaggle_index,gaggles, max_k, id_length);
		gaggle_index += gaggles;
		++gaggle_id;
	}
	// Additional bit planes opf DC coefficients 4.3.3
	if (header_pointer->header.part1.bit_depth_ac_5bits < quantization_factor)
	{
		int num_add_bit_planes;
		if (header_pointer->header.part4.dwt_type == INTEGER_WAVELET)
		{
			num_add_bit_planes = quantization_factor - (header_pointer ->header.part1.bit_depth_ac_5bits > header_pointer->header.part4.custom_wt_LL3_2bits ? header_pointer ->header.part1.bit_depth_ac_5bits : header_pointer->header.part4.custom_wt_LL3_2bits); // max operation (づ｡◕‿‿◕｡)づ
		}
		else
		{
			num_add_bit_planes = quantization_factor - header_pointer->header.part1.bit_depth_ac_5bits;
		}
		list_of_process_data[block_position/SEGMENTSIZE].numaddbitplanes = num_add_bit_planes;
		list_of_process_data[block_position/SEGMENTSIZE].dc_remainer = dc_remainer;
		// loop over the remainders to print

	}
	

}


void dc_encoder(unsigned int gaggle_id, unsigned int block_position,block_struct* list_of_process_data, header_struct *header_pointer, unsigned long *dc_mapper, short N, int star_index, int gaggles, int max_k, int id_length)
{
	
	int min_k = 0;
	unsigned long min_bits = 0xFFFF;
	unsigned long total_bits = 0;

	unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
	// determine code choice min_k to use, either via brute-force optimum calculation, or heuristic approach
	// 4.3.2.11
	if (header_pointer->header.part3.opt_dc_select == TRUE) // Brute force (╯°□°）╯︵ ┻━┻ 
	{
		min_k = uncoded_flag;  // uncoded option used, unless we find a better option below
		for(unsigned int k = 0; k <= max_k; ++k)
		{
			if (star_index == 0)
			{
				total_bits = N;
			}
			else
			{
				total_bits = 0;
			}
			int max_value = star_index > 1 ? star_index : 1; // max operation (づ｡◕‿‿◕｡)づ
			for (unsigned int i = max_value; i < star_index + gaggles; ++i)
			{
				total_bits += ((dc_mapper[i] >> k) + 1) + k; // coded sample cost
			}
			if((total_bits < min_bits) && (total_bits < (N * gaggles)))
			{
				min_bits = total_bits;
				min_k = k;
			}
		}
		
	}
	else // heuristic code option selection (っ▀¯▀)つ
	{
		int delta = 0;
		int j = gaggles;
		if (star_index == 0)
		{
			j = gaggles -1;
		}
		for (unsigned int i = star_index; i < star_index + gaggles; ++i)
		{
			delta += dc_mapper[i];
		}
		// Table 4-10
		if (64*delta >= 23*j*(1 << N))
		{
			min_k = uncoded_flag; // indicate that uncoded option is to be used
		}
		else if (207 * j > 128 * delta)
		{
			min_k = 0;
		}
		else if ((long)(j * (1<< (N + 5)))  <= (long)(128 * delta + 49 * j))
		{
			min_k = N -2;
		}
		else
		{
			min_k = 0;
			// ¯\\_(ツ)_/¯ 
			while ((long)(j * (1 << (min_k + 7))) <= (long)(128 - delta + 49 *j) )
			{
				++min_k; // ¯\\_(ツ)_/¯ 
			}
			--min_k; // ¯\\_(ツ)_/¯ 
			
		}
	}
	list_of_process_data[block_position/SEGMENTSIZE].min_k[gaggle_id] = min_k;
	// Print bits min_k
	
}
// bit procssing time
void  set_bit(unsigned char A[],  int k )
{
    A[k/8] |= 1 << (k%8);  // Set the bit at the k-th position in A[i]
}

void  clear_bit(unsigned char A[],  int k )                
{
    A[k/8] &= ~(1 << (k%8));
}

int test_bit(unsigned char A[],  int k )
{
    return ( (A[k/8] & (1 << (k%8) )) != 0 ) ;     
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned char value, unsigned int intial_position, unsigned int size)
{   

    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
            clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned long value, unsigned int intial_position, unsigned int size)
{
    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
           clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

unsigned int write_data_to_array(unsigned char *data_array, unsigned short value, unsigned int intial_position, unsigned int size)
{
    for (unsigned int i = 0; i < size;++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((size -1) - i) + intial_position);
        }
        else
        {
           clear_bit(data_array, ((size -1) - i) + intial_position);
        }
    }
	return intial_position + size;
}

void swap_char(unsigned char *data_array, unsigned char value, unsigned int array_pos)
{   
    
     for (unsigned int i = 0; i < 8; ++i)
    {
        if (((value >> i) & 0x01 ) == 1)
        {
            set_bit(data_array, ((8 -1) - i) + array_pos * 8);
        }
        else
        {
            clear_bit(data_array, ((8 -1) - i) + array_pos * 8);
        }
    }

}

void read_array(FILE *ofp, unsigned char *data_array, unsigned int size, unsigned int array_size)
{

    for (unsigned int i = 0; i < array_size; ++i)
    {   
        swap_char(data_array,data_array[i], i);
        fprintf(ofp, "%02X ",data_array[i]);
    }
}


void convert_to_bits_segment(FILE *ofp, block_struct segment_data)
{
	std::string data_out = "";
	unsigned int total_bits = 0;
	// get the total bits needed
	// bits in header 
	total_bits += 160;
	unsigned int gaggle_index = 0;
	unsigned int gaggle_id = 0;
	unsigned int gaggles = 0;
	// bits in shifted_dc
	if (segment_data.N == 1)
	{
		total_bits += segment_data.header->part3.seg_size_blocks_20bits; // shifted DC is size 1 bit for numer of blocks so number of blocks
	}
	else
	{
		// DC encoder
		while(gaggle_index < segment_data.header->part3.seg_size_blocks_20bits)
		{
			gaggles = GAGGLE_SIZE > (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) ? (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
			
			// min K
			total_bits += segment_data.bits_min_k; // add one per gaggle in the block list
			unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
			// mappedDC
				for (unsigned int i = gaggle_index; i < gaggle_index + gaggles; ++i)
				{
					if ((segment_data.min_k[gaggle_id] == uncoded_flag) || (i==0))
					{
						total_bits += segment_data.N;
					}
					else
					{
						total_bits +=(segment_data.dc_mapper[i] >> segment_data.min_k[gaggle_id]) +1;
					}
				}
				// if we have coded samples, then we also have to send the second part
				if (segment_data.min_k[gaggle_id] != uncoded_flag)
				{	
					int max_value = gaggle_index > 1 ? gaggle_index : 1; // max operation (づ｡◕‿‿◕｡)づ
					for(unsigned int i = max_value; i < gaggle_index + gaggles; ++i)
					{
						// print_out dc_mapper[i], min_k
						total_bits += segment_data.min_k[gaggle_id];
					}
				}
			gaggle_index += gaggles;
			gaggle_id += 1;
		}
	}
	// bits DC renaning
	total_bits += segment_data.numaddbitplanes * segment_data.header->part3.seg_size_blocks_20bits;
	// total bits calculate
	unsigned int array_size = ceil(float(total_bits)/float(8));
	//unsigned int last_position_number_bits = total_bits % 8;
	// declarate data array
	unsigned char *data_array = NULL;
	data_array = (unsigned char*)calloc(array_size, sizeof(unsigned char));
	
	// print header
	// part 1
	unsigned int position = 0;
	position = write_data_to_array(data_array, segment_data.header->part1.start_img_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.end_img_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.segment_count_8bits ,position,8);
	position = write_data_to_array(data_array, segment_data.header->part1.bit_depth_dc_5bits ,position,5);
	position = write_data_to_array(data_array, segment_data.header->part1.bit_depth_ac_5bits ,position,5);
	position = write_data_to_array(data_array, segment_data.header->part1.reserved ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_2_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_3_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.part_4_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part1.pad_rows_3bits ,position,3);
	position = write_data_to_array(data_array, segment_data.header->part1.reserved_5bits ,position,5);

	// part 2
	position = write_data_to_array(data_array, segment_data.header->part2.seg_byte_limit_27bits ,position,27);
	position = write_data_to_array(data_array, segment_data.header->part2.dc_stop ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.bit_plane_stop_5bits ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.stage_stop_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part2.use_fill ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part2.reserved_4bits ,position,4);

	// part 3
	position = write_data_to_array(data_array, segment_data.header->part3.seg_size_blocks_20bits ,position,20);
	position = write_data_to_array(data_array, segment_data.header->part3.opt_dc_select ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part3.opt_ac_select ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part3.reserved_2bits ,position,2);

	// part 3
	position = write_data_to_array(data_array, segment_data.header->part4.dwt_type ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.reserved_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.signed_pixels ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.pixel_bit_depth_4bits ,position,4);
	position = write_data_to_array(data_array, segment_data.header->part4.image_with_20bits ,position,20);
	position = write_data_to_array(data_array, segment_data.header->part4.transpose_img ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.code_word_length ,position,3);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_flag ,position,1);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH1_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH2_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HH3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_HL3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LH3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.custom_wt_LL3_2bits ,position,2);
	position = write_data_to_array(data_array, segment_data.header->part4.reserved_11bits ,position,11);
	// print shifted_dc
	if (segment_data.N == 1)
	{
		for (unsigned int i = 0; i < segment_data.header->part3.seg_size_blocks_20bits; ++i )
		{
			position = write_data_to_array(data_array, segment_data.shifted_dc_1bit[i] ,position,1);
		}
	}
	else
	{
		// pint DC encoder
		gaggle_index = 0;
		gaggle_id = 0;
		gaggles = 0;
		while(gaggle_index < segment_data.header->part3.seg_size_blocks_20bits)
		{
			gaggles = GAGGLE_SIZE > (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) ? (segment_data.header->part3.seg_size_blocks_20bits - gaggle_index) : GAGGLE_SIZE; // min operation (づ｡◕‿‿◕｡)づ
			
			// min K
			total_bits += segment_data.bits_min_k; // add one per gaggle in the block list
			unsigned char uncoded_flag = ~0; // this is the flag indicating that uncoded option used
			// mappedDC
				for (unsigned int i = gaggle_index; i < gaggle_index + gaggles; ++i)
				{
					if ((segment_data.min_k[gaggle_id] == uncoded_flag) || (i==0))
					{
						position = write_data_to_array(data_array, segment_data.dc_mapper[i] ,position,segment_data.N);
					}
					else
					{
						position = write_data_to_array(data_array, (unsigned char)1 ,position,(segment_data.dc_mapper[i] >> segment_data.min_k[gaggle_id]) +1);
						
					}
				}
				// if we have coded samples, then we also have to send the second part
				if (segment_data.min_k[gaggle_id] != uncoded_flag)
				{	
					int max_value = gaggle_index > 1 ? gaggle_index : 1; // max operation (づ｡◕‿‿◕｡)づ
					for(unsigned int i = max_value; i < gaggle_index + gaggles; ++i)
					{
						// print_out dc_mapper[i], min_k
						position = write_data_to_array(data_array, segment_data.dc_mapper[i] ,position,segment_data.min_k[gaggle_id]);
					}
				}
			gaggle_index += gaggles;
			gaggle_id += 1;
		}
	}
	for (unsigned int i = 0; i < segment_data.numaddbitplanes; ++i)
	{
		for(unsigned int k = 0; k < segment_data.header->part3.seg_size_blocks_20bits; ++k)
		{
			position = write_data_to_array(data_array, (unsigned char)(segment_data.dc_remainer[k] >> (segment_data.quantization_factor - i -1) ),position,1);
		}
	}


	// print the array
	read_array(ofp, data_array,total_bits, array_size);



}

unsigned long conversion_twos_complement(long original, short leftmost)
{

	unsigned long temp;
	unsigned long complement;

	short i = 0;
	if (leftmost == 1)
	{
		return 0;
	}
	
	if (leftmost >= sizeof(unsigned long) * 8 || (leftmost == 0))
	{
		return 0; // ERROR IN THE DATA
	}

	if (original >=0)
	{
		return (unsigned long) original;
	}
	else
	{
		complement = ~(unsigned long) (-original);
		temp = 0;
		for (i = 0; i < leftmost; ++i)
		{
			temp <<= 1;
			++temp;
		}
		complement &= temp;
		complement ++;
		return complement;
	}
	
}

void syncstreams(){
    for (unsigned int x = 0; x < NUMBER_STREAMS; ++x) {queues[x].finish();}
}

void clean(DataObject *device_object){
}