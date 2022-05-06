/**
 * \file device.c
 * \brief Benchmark #3.1 CPU version (sequential) device initialization. 
 * \author Marc Sole(BSC)
 */
#include "benchmark.h"
#include "device.h"
#include "GEN_processing.hcl"
#include <iostream>

void init(
	AES_data_t *AES_data,
	AES_time_t *t,
	char *device_name
	)
{
    init(AES_data,t, 0,0, device_name);
}

void init(
	AES_data_t *AES_data,
	AES_time_t *t,
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
    AES_data->context = new cl::Context(default_device);
    AES_data->queue = new cl::CommandQueue(*AES_data->context,default_device,NULL);
    AES_data->default_device = default_device;

    // events
    //t->t_device_host = new cl::Event();

    // program
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_def_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    AES_data->program = new cl::Program(*AES_data->context,sources);
    if(AES_data->program->build({AES_data->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<AES_data->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(AES_data->default_device)<<"\n";
        exit(1);
    }

}


bool device_memory_init(
	AES_data_t *AES_data,
    unsigned int key_size,
    unsigned int data_length
	)
{	
	/* key configuration values initialization */
    AES_data->key = (AES_key_t*) malloc(sizeof(AES_key_t));
	AES_data->key->size = (AES_keysize_t) key_size;
	switch(key_size) {
		case AES_KEY128: AES_data->key->Nk = 4; AES_data->key->Nr = 10; break;
		case AES_KEY192: AES_data->key->Nk = 6; AES_data->key->Nr = 12; break;
		case AES_KEY256: AES_data->key->Nk = 8; AES_data->key->Nr = 14; break;
	}
	AES_data->key->Nb = 4;

	/* key value memory allocation */
	AES_data->key->value = new cl::Buffer(*AES_data->context, CL_MEM_READ_WRITE, sizeof(uint8_t)*key_size);

	/* memory allocation for input and output texts */
    AES_data->plaintext = new cl::Buffer(*AES_data->context, CL_MEM_READ_ONLY, sizeof(uint8_t)*data_length);
    AES_data->cyphertext = new cl::Buffer(*AES_data->context, CL_MEM_READ_WRITE,sizeof(uint8_t)*data_length);
    AES_data->iv = new cl::Buffer(*AES_data->context, CL_MEM_READ_WRITE,sizeof(uint8_t)*data_length);
    AES_data->data_length = data_length;

    /* allocate constant lookup tables */
    /* memory for sbox (256 uint8) */
    AES_data->sbox = new cl::Buffer(*AES_data->context, CL_MEM_READ_ONLY, sizeof(unsigned char)*256);
    /* memory for rcon (11 uint8) */
    AES_data->rcon = new cl::Buffer(*AES_data->context, CL_MEM_READ_ONLY, sizeof(unsigned char)*11);

    /* memory for roundkey (expanded key Nb*(Nr+1) uint32) */
    AES_data->expanded_key = new cl::Buffer(*AES_data->context, CL_MEM_READ_WRITE,sizeof(unsigned char)*4*AES_data->key->Nb*(AES_data->key->Nr+1));

    return true;
}

void copy_memory_to_device(
	AES_data_t *AES_data,
	AES_time_t *t,
	uint8_t *input_key,
	uint8_t *input_text,
	uint8_t *input_iv,
	uint8_t *input_sbox,
	uint8_t *input_rcon
	)
{
    T_START(t->t_host_device);
    /* initialize key value */
    AES_data->queue->enqueueWriteBuffer(*AES_data->key->value, CL_TRUE, 0, sizeof(unsigned char)*AES_data->key->size/8, input_key, NULL, NULL);
    /* initialize input text */
    AES_data->queue->enqueueWriteBuffer(*AES_data->plaintext, CL_TRUE, 0, sizeof(unsigned char)*AES_data->data_length, input_text, NULL, NULL);
    /* initialize initialization vector */
    AES_data->queue->enqueueWriteBuffer(*AES_data->iv, CL_TRUE, 0, sizeof(unsigned char)*16, input_iv, NULL, NULL);
    /* initialize sbox */
    AES_data->queue->enqueueWriteBuffer(*AES_data->sbox, CL_TRUE, 0, sizeof(unsigned char)*256, input_sbox, NULL, NULL);
    /* initialize rcon */
    AES_data->queue->enqueueWriteBuffer(*AES_data->rcon, CL_TRUE, 0, sizeof(unsigned char)*11, input_rcon, NULL, NULL);
    T_STOP(t->t_host_device);
}


void process_benchmark(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{    
    int n_blocks = AES_data->data_length/ (4*AES_data->key->Nb);
    cl::NDRange global_range;
    global_range = cl::NDRange(n_blocks);
    cl::NDRange one_range;
    one_range = cl::NDRange(1);
    
    T_START(t->t_test);
    cl::Kernel kernel_expansion=cl::Kernel(*AES_data->program, "AES_KeyExpansion");
    kernel_expansion.setArg(0, AES_data->key->Nb);
    kernel_expansion.setArg(1, AES_data->key->Nr);
    kernel_expansion.setArg(2, AES_data->key->Nk);
    kernel_expansion.setArg(3, *AES_data->key->value);
    kernel_expansion.setArg(4, *AES_data->expanded_key);
    kernel_expansion.setArg(5, *AES_data->sbox);
    kernel_expansion.setArg(6, *AES_data->rcon);
    AES_data->queue->enqueueNDRangeKernel(kernel_expansion, cl::NullRange, one_range, cl::NullRange, NULL, NULL);
    AES_data->queue->finish();
    cl::Kernel kernel_encrypt=cl::Kernel(*AES_data->program, "AES_encrypt");
    kernel_encrypt.setArg(0, *AES_data->plaintext);
    kernel_encrypt.setArg(1, *AES_data->cyphertext);
    kernel_encrypt.setArg(2, *AES_data->iv);
    kernel_encrypt.setArg(3, AES_data->key->Nb);
    kernel_encrypt.setArg(4, AES_data->key->Nr);
    kernel_encrypt.setArg(5, *AES_data->sbox);
    kernel_encrypt.setArg(6, *AES_data->expanded_key);
    AES_data->queue->enqueueNDRangeKernel(kernel_encrypt, cl::NullRange, global_range, cl::NullRange, NULL, NULL);
    AES_data->queue->finish();
    T_STOP(t->t_test);
}

void copy_memory_to_host(
	AES_data_t *AES_data,
	AES_time_t *t,
	uint8_t *output
	)
{
    //FIXME Switch to openCL timing when verify in other platform
    T_START(t->t_device_host);
    AES_data->queue->enqueueReadBuffer(*AES_data->cyphertext, CL_TRUE, 0, sizeof(unsigned char)*AES_data->data_length, output, NULL, NULL); //t->t_device_host);
    T_STOP(t->t_device_host);
}


void get_elapsed_time(
	AES_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{	
    //t->t_device_host->wait();
    
    double host_to_device = (t->t_host_device)/((double) (CLOCKS_PER_SEC / 1000));
    double elapsed_time = (t->t_test)/((double) (CLOCKS_PER_SEC / 1000));
    double device_to_host = (t->t_device_host)/((double) (CLOCKS_PER_SEC / 1000)); 
    //double device_to_host = t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_START>();

	if (csv_format)
	{
		printf("%.10f;%.10f;%.10f;\n", host_to_device, elapsed_time, device_to_host);
	}
	else if (database_format)
	{
		printf("%.10f;%.10f;%.10f;%ld;\n", host_to_device, elapsed_time, device_to_host, timestamp);
	}
	else if(verbose_print)
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", host_to_device);
		printf("Elapsed time kernel: %.10f milliseconds\n", elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", device_to_host);
	}
}


void clean(
	AES_data_t *AES_data,
	AES_time_t *t
	)
{
	/* Clean time */
	free(AES_data->key);
}
