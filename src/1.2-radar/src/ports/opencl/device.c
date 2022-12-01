/**
 * \file device.c
 * \brief Benchmark #1.2 GPU version (cuda) device initialization. 
 * \author Marc Sole Bonet (BSC)
 */
#include "benchmark.h"
#include "obpmark_time.h"
#include "device.h"
#include "GEN_processing.hcl"

uint32_t next_power_of_two(uint32_t n)
{
    uint32_t v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void init(
	radar_data_t *radar_data,
	radar_time_t *t,
	char *device_name
	)
{
    init(radar_data,t, 0,0, device_name);
}


void init(
	radar_data_t *radar_data,
	radar_time_t *t,
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
    radar_data->context = new cl::Context(default_device);
    radar_data->queue = new cl::CommandQueue(*radar_data->context,default_device,NULL);
    radar_data->default_device = default_device;

    // events
    //t->t_device_host = new cl::Event();

    // program
    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_def_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});

    radar_data->program = new cl::Program(*radar_data->context,sources);
    if(radar_data->program->build({radar_data->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<radar_data->program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(radar_data->default_device)<<"\n";
        exit(1);
    }

}


bool device_memory_init(
	radar_data_t *radar_data,
	radar_params_t *params,
    unsigned int out_height,
    unsigned int out_width
	)
{	
    unsigned int patch_width = params->rsize<<1;
    unsigned int patch_extended_width = next_power_of_two(params->rsize)<<1;
    unsigned int patch_height = params->apatch;

    radar_data->out_height = out_height;
    radar_data->out_width = out_width;
    radar_data->params = params;

    float z = 0;

	/* radar_data_t memory allocation */
	//RANGE & AZIMUTH DATA
	radar_data->range_data = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(float) * params->npatch * patch_height * patch_extended_width);
    radar_data->queue->enqueueFillBuffer(*radar_data->range_data, z, 0,
	        sizeof(float) * params->npatch * patch_height * patch_extended_width, NULL, NULL);


	radar_data->azimuth_data = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(float) * params->npatch * patch_height * patch_width);
    radar_data->queue->enqueueFillBuffer(*radar_data->azimuth_data, z, 0,
	        sizeof(float) * params->npatch * patch_height * patch_width, NULL, NULL);

  	//MULTILOOK DATA
	radar_data->ml_data = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(float) * params->npatch * out_height * out_width);

  	//OUTPUT DATA
	radar_data->output_image = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(uint8_t) * params->npatch * out_height * out_width);

    //RANGE REF. FUNCTION
	radar_data->rrf = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(float) * patch_extended_width);
    radar_data->queue->enqueueFillBuffer(*radar_data->rrf, z, 0,
	        sizeof(float) * patch_extended_width,  NULL, NULL);

	//AZIMUTH REF. FUNCTION
	radar_data->arf = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
	        sizeof(float) * (patch_height<<1));
    radar_data->queue->enqueueFillBuffer(*radar_data->arf, z, 0,
	        sizeof(float) * (patch_height<<1),  NULL, NULL);

    //RCMC TABLE
	radar_data->offsets = new cl::Buffer(*radar_data->context, CL_MEM_READ_WRITE,
            sizeof(uint32_t) * params->rvalid * patch_height);

    return true;
}

void copy_memory_to_device(
	radar_data_t *radar_data,
	radar_time_t *t,
	framefp_t *input_data,
	radar_params_t *input_params
	)
{
    T_START(t->t_host_device);
    /* Copy params */
    uint32_t width = input_params->rsize<<1;
    uint32_t height = input_params->apatch;
    uint32_t line_width = next_power_of_two(width);
    uint32_t patch_size = line_width * height;
    for (uint32_t i = 0; i < input_params->npatch; i++ )
        for(uint32_t j = 0; j < height; j++){
            uint32_t offs = (i * patch_size + j * line_width) * sizeof(float);
            radar_data->queue->enqueueWriteBuffer(*radar_data->range_data, CL_TRUE, offs,
                    width * sizeof(float), &input_data[i].f[j*width], NULL, NULL);
        }

    T_STOP(t->t_host_device);
}

const int FFT_FORWARD = 1;
const int FFT_INVERSE = -1;

void launch_fft(cl::Kernel fft_kernel, cl::Kernel bin_rev_kernel, cl::CommandQueue *queue, cl::Buffer *data, unsigned int width, unsigned int rows, unsigned int npatch, int direction)
{
    unsigned int group = (unsigned int) log2(width);
    bin_rev_kernel.setArg(0, *data);
    bin_rev_kernel.setArg(1, width);
    bin_rev_kernel.setArg(2, group);
    queue->enqueueNDRangeKernel(bin_rev_kernel, cl::NullRange, cl::NDRange(width, rows * npatch), cl::NullRange, NULL, NULL);

    unsigned int nthreads = width>>1;
    float wtemp, wpr, wpi, theta;
    int loop = 1;

    while(loop < width){
        // calculate values
        theta = -M_PI/(loop*direction); // check
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        queue->finish();
        //kernel launch
        fft_kernel.setArg(0, *data);
        fft_kernel.setArg(1, loop);
        fft_kernel.setArg(2, wpr);
        fft_kernel.setArg(3, wpi);
        fft_kernel.setArg(4, nthreads);
        fft_kernel.setArg(5, width);
        queue->enqueueNDRangeKernel(fft_kernel, cl::NullRange, cl::NDRange(nthreads, rows * npatch), cl::NullRange, NULL, NULL);
        // update loop values
        loop = loop * 2;
    }
    queue->finish();
}

void process_benchmark(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{    
    radar_params_t *params = radar_data->params;

    T_START(t->t_test);

    cl::Kernel fft_kernel=cl::Kernel(*radar_data->program, "fft_kernel");
    cl::Kernel bin_rev_kernel=cl::Kernel(*radar_data->program, "bin_reverse");

    unsigned int nit = floor(params->tau * params->fs);   
    /* RANGE REFERENCE */
    cl::Kernel rrf_kernel=cl::Kernel(*radar_data->program, "SAR_range_ref");
    rrf_kernel.setArg(0, *radar_data->rrf);
    rrf_kernel.setArg(1, params->rsize);
    rrf_kernel.setArg(2, params->fs);
    rrf_kernel.setArg(3, params->slope);
    rrf_kernel.setArg(4, nit);
    radar_data->queue->enqueueNDRangeKernel(rrf_kernel, cl::NullRange, cl::NDRange(params->rsize), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->rrf, next_power_of_two(params->rsize), 1, 1, FFT_FORWARD);


    /* DOPPLER CENTROID */
    cl::Kernel DCE_kernel=cl::Kernel(*radar_data->program, "SAR_DCE");
    DCE_kernel.setArg(0, *radar_data->range_data);
    DCE_kernel.setArg(1, params->apatch);
    DCE_kernel.setArg(2, params->rsize);
    DCE_kernel.setArg(3, params->PRF/(2*pi*params->rsize));
    DCE_kernel.setArg(4, sizeof(float)*2*params->apatch, NULL);
    radar_data->queue->enqueueNDRangeKernel(DCE_kernel, cl::NullRange, cl::NDRange(params->rsize*BLOCK_SIZE), cl::NDRange(BLOCK_SIZE), NULL, NULL);
    radar_data->queue->finish();

    /* RCMC table */
    cl::Kernel off_kernel = cl::Kernel(*radar_data->program, "SAR_rcmc_table");
    off_kernel.setArg(0, *radar_data->offsets);
    off_kernel.setArg(1, params->apatch);
    off_kernel.setArg(2, params->avalid);
    off_kernel.setArg(3, params->PRF);
    off_kernel.setArg(4, params->lambda);
    off_kernel.setArg(5, params->vr);
    off_kernel.setArg(6, params->ro);
    off_kernel.setArg(7, params->fs);
    radar_data->queue->enqueueNDRangeKernel(off_kernel, cl::NullRange, cl::NDRange(params->apatch, params->rvalid), cl::NullRange, NULL, NULL);

    /* AZIMUTH REFERENCE */
    cl::Kernel arf_kernel=cl::Kernel(*radar_data->program, "SAR_azimuth_ref");
    arf_kernel.setArg(0, *radar_data->arf);
    arf_kernel.setArg(1, params->ro);
    arf_kernel.setArg(2, params->fs);
    arf_kernel.setArg(3, params->lambda);
    arf_kernel.setArg(4, params->vr);
    arf_kernel.setArg(5, params->PRF);
    arf_kernel.setArg(6, params->rvalid);
    arf_kernel.setArg(7, params->apatch);
    radar_data->queue->enqueueNDRangeKernel(arf_kernel, cl::NullRange, cl::NDRange(params->apatch), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->arf, params->apatch, 1, 1, FFT_FORWARD);

    /* RANGE COMPRESS */
    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->range_data, next_power_of_two(params->rsize), params->apatch, params->npatch, FFT_FORWARD);

    cl::Kernel ref_kernel=cl::Kernel(*radar_data->program, "SAR_ref_product");
    ref_kernel.setArg(0, *radar_data->range_data);
    ref_kernel.setArg(1, *radar_data->rrf);
    ref_kernel.setArg(2, next_power_of_two(params->rsize));
    ref_kernel.setArg(3, params->apatch);
    radar_data->queue->enqueueNDRangeKernel(ref_kernel, cl::NullRange, cl::NDRange(next_power_of_two(params->rsize), params->apatch, params->npatch), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->range_data, next_power_of_two(params->rsize), params->apatch, params->npatch, FFT_INVERSE);


    cl::Kernel trans_kernel=cl::Kernel(*radar_data->program, "SAR_transpose");
    trans_kernel.setArg(0, *radar_data->range_data);
    trans_kernel.setArg(1, *radar_data->azimuth_data);
    trans_kernel.setArg(2, next_power_of_two(params->rsize));
    trans_kernel.setArg(3, params->apatch);
    trans_kernel.setArg(4, params->apatch);
    trans_kernel.setArg(5, params->rvalid);
    radar_data->queue->enqueueNDRangeKernel(trans_kernel, cl::NullRange, cl::NDRange(next_power_of_two(params->rsize), params->apatch, params->npatch), cl::NDRange(TILE_SIZE, TILE_SIZE, 1), NULL, NULL);
    radar_data->queue->finish();

    /* AZIMUTH COMPRESS */
    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->azimuth_data, params->apatch, params->rvalid, params->npatch, FFT_FORWARD);

    cl::Kernel rcmc_kernel=cl::Kernel(*radar_data->program, "SAR_rcmc");
    rcmc_kernel.setArg(0, *radar_data->azimuth_data);
    rcmc_kernel.setArg(1, *radar_data->offsets);
    rcmc_kernel.setArg(2, params->apatch);
    rcmc_kernel.setArg(3, params->rvalid);
    radar_data->queue->enqueueNDRangeKernel(rcmc_kernel, cl::NullRange, cl::NDRange(params->apatch, params->npatch, 1), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    ref_kernel.setArg(0, *radar_data->azimuth_data);
    ref_kernel.setArg(1, *radar_data->arf);
    ref_kernel.setArg(2, params->apatch);
    ref_kernel.setArg(3, params->rvalid);
    radar_data->queue->enqueueNDRangeKernel(ref_kernel, cl::NullRange, cl::NDRange(params->apatch, params->rvalid, params->npatch), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    launch_fft(fft_kernel, bin_rev_kernel, radar_data->queue, radar_data->azimuth_data, params->apatch, params->rvalid, params->npatch, FFT_INVERSE);

    trans_kernel.setArg(0, *radar_data->azimuth_data);
    trans_kernel.setArg(1, *radar_data->range_data);
    trans_kernel.setArg(2, params->apatch);
    trans_kernel.setArg(3, next_power_of_two(params->rsize));
    trans_kernel.setArg(4, params->rvalid);
    trans_kernel.setArg(5, params->apatch);
    radar_data->queue->enqueueNDRangeKernel(trans_kernel, cl::NullRange, cl::NDRange(params->apatch, params->rvalid, params->npatch), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    /* IMAGE PROCESSING */
    cl::Kernel ml_kernel=cl::Kernel(*radar_data->program, "SAR_multilook");
    ml_kernel.setArg(0, *radar_data->range_data);
    ml_kernel.setArg(1, *radar_data->ml_data);
    ml_kernel.setArg(2, params->rvalid);
    ml_kernel.setArg(3, params->asize);
    ml_kernel.setArg(4, params->rsize);
    ml_kernel.setArg(5, params->npatch);
    ml_kernel.setArg(6, params->apatch);
    ml_kernel.setArg(7, radar_data->out_width);
    ml_kernel.setArg(8, radar_data->out_height);
    radar_data->queue->enqueueNDRangeKernel(ml_kernel, cl::NullRange, cl::NDRange(radar_data->out_height, radar_data->out_width), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    cl::Kernel q_kernel=cl::Kernel(*radar_data->program, "quantize");
    q_kernel.setArg(0, *radar_data->ml_data);
    q_kernel.setArg(1, *radar_data->output_image);
    q_kernel.setArg(2, radar_data->out_width);
    q_kernel.setArg(3, radar_data->out_height);
    radar_data->queue->enqueueNDRangeKernel(q_kernel, cl::NullRange, cl::NDRange(radar_data->out_height, radar_data->out_width), cl::NullRange, NULL, NULL);
    radar_data->queue->finish();

    T_STOP(t->t_test);
}

void copy_memory_to_host(
	radar_data_t *radar_data,
	radar_time_t *t,
	frame8_t *output_radar
	)
{
    T_START(t->t_device_host);
    uint32_t  width = output_radar->w;
    uint32_t  height = output_radar->h;
    radar_data->queue->enqueueReadBuffer(*radar_data->output_image, CL_TRUE, 0,
            sizeof(uint8_t) * width * height, output_radar->f, NULL, NULL);
    T_STOP(t->t_device_host);
}


void get_elapsed_time(
	radar_data_t *radar_data, 
	radar_time_t *t, 
    print_info_data_t *benchmark_info.c,
	long int timestamp
	)
{	
    //t->t_device_host->wait();
    
    double host_to_device = (t->t_host_device)/((double) (CLOCKS_PER_SEC / 1000));
    double elapsed_time = (t->t_test)/((double) (CLOCKS_PER_SEC / 1000));
    double device_to_host = (t->t_device_host)/((double) (CLOCKS_PER_SEC / 1000)); 
    //double device_to_host = t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_END>() - t->t_device_host->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    print_execution_info(benchmark_info, true, timestamp,host_to_device,elapsed_time,device_to_host);
}


void clean(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{

	/* Clean time */
	free(t);
    delete radar_data->program;
    delete radar_data->context;
    delete radar_data->queue;

    delete radar_data->range_data; //width: range, height: azimuth
    delete radar_data->azimuth_data; //width: azimuth, height: range
    delete radar_data->ml_data;
    delete radar_data->output_image;
    delete radar_data->rrf; //range reference function
    delete radar_data->arf; // azimuth reference function
    delete radar_data->offsets; //Offset table for RCMC
}
