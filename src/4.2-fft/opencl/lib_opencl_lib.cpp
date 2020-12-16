// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <chrono>
#include <clFFT.h>


//#define BLOCK_SIZE 32
void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}
void init(GraficObject *device_object, int platform ,int device, char* device_name){
    //get all platforms (drivers)
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(all_platforms.size()==0){
        std::cout<<" No platforms found. Check OpenCL installation!\n";
        exit(1);
    }
    cl::Platform default_platform=all_platforms[platform];
    //std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
   //get default device of the default platform
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
    device_object->evt = new cl::Event; 
    device_object->evt_copyB = new cl::Event;
    device_object->evt_copyBr = new cl::Event;
    

}

bool device_memory_init(GraficObject *device_object, int64_t size){
   device_object->d_B = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size);
   device_object->d_Br = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size);
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size){
	// copy memory host -> device
	//TODO Errors check
    device_object->queue->enqueueWriteBuffer(*device_object->d_B,CL_TRUE,0,sizeof(bench_t)*size, h_B, NULL, device_object->evt_copyB);
}

void execute_kernel(GraficObject *device_object, int64_t size){
    struct timespec start, end;
    const unsigned int x_local= BLOCK_SIZE;
    unsigned int mode = (unsigned int)log2(size);

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);

    clfftPlanHandle planHandle;
    size_t clLengths[1] = {size};
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    clfftCreateDefaultPlan(&planHandle, (*device_object->context)(), CLFFT_1D, clLengths);

    /* Set plan parameters. */
    #ifdef FLOAT
    clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
    #else
    clfftSetPlanPrecision(planHandle, CLFFT_DOUBLE);
    #endif
    clfftSetLayout(planHandle, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(planHandle, CLFFT_INPLACE);

    /* Bake the plan. */
    clfftBakePlan(planHandle, 1, &(*device_object->queue)(), NULL, NULL);

    /* Execute the plan. */
    clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &(*device_object->queue)(), 0,NULL, &(*device_object->evt)(), &(*device_object->d_B)(), NULL, NULL);

    device_object->queue->finish();
    clfftDestroyPlan( &planHandle );
    clfftTeardown( );
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    device_object->elapsed_time =  (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
   
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size){
    device_object->queue->enqueueReadBuffer(*device_object->d_B,CL_TRUE,0,sizeof(bench_t)*size,h_B, NULL, device_object->evt_copyBr);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    device_object->evt_copyBr->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed = device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    elapsed_d_h = device_object->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyBr->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );


    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,device_object->elapsed_time,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f miliseconds\n", device_object->elapsed_time);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficObject *device_object){
    // pointers clean
    delete device_object->context;
    delete device_object->queue;
    // pointer to memory
    delete device_object->d_B;
    delete device_object->d_Br;
    delete device_object->evt;
    delete device_object->evt_copyB;
    delete device_object->evt_copyBr;
}
