// OpenCL lib code 
#include <cmath>
#include "../benchmark_library.h"
#include <cstring>
#include <string>
#include "GEN_kernel_opt.hcl"


//#define BLOCK_SIZE 16
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
    device_object->evt_copyA = new cl::Event;
    device_object->evt_copyB = new cl::Event;
    device_object->evt_copyC = new cl::Event;
    
}

bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){
   device_object->d_A = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_a_matrix);
   device_object->d_B = new cl::Buffer(*device_object->context,CL_MEM_READ_ONLY ,sizeof(bench_t)*size_b_matrix);
   device_object->d_C = new cl::Buffer(*device_object->context,CL_MEM_READ_WRITE ,sizeof(bench_t)*size_c_matrix);
   // inicialice Arrays
   return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b){
	// copy memory host -> device
	//TODO Errors check
    device_object->queue->enqueueWriteBuffer(*device_object->d_A,CL_TRUE,0,sizeof(bench_t)*size_a, h_A, NULL, device_object->evt_copyA);
    device_object->queue->enqueueWriteBuffer(*device_object->d_B,CL_TRUE,0,sizeof(bench_t)*size_b, h_B, NULL, device_object->evt_copyB);
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w){
    const unsigned int x_local= BLOCK_SIZE;
    const unsigned int y_local= BLOCK_SIZE;
    cl::NDRange local(x_local, y_local);
    cl::NDRange global(n, w);

    cl::Program::Sources sources;
    // load kernel from file
    kernel_code = type_kernel + kernel_code;
    sources.push_back({kernel_code.c_str(),kernel_code.length()});
    cl::Program program(*device_object->context,sources);
    if(program.build({device_object->default_device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_object->default_device)<<"\n";
        exit(1);
    }
    cl::Kernel kernel_add=cl::Kernel(program,"kernel_matrix_multiplication");
    kernel_add.setArg(0,*device_object->d_A);
    kernel_add.setArg(1,*device_object->d_B);
    kernel_add.setArg(2,*device_object->d_C);
    kernel_add.setArg(3,n);
    kernel_add.setArg(4,m);
    kernel_add.setArg(5,w);
    kernel_add.setArg(6,BLOCK_SIZE);

    device_object->queue->enqueueNDRangeKernel(kernel_add,cl::NullRange,global,local, NULL, device_object->evt);
    device_object->queue->finish();

}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
	device_object->queue->enqueueReadBuffer(*device_object->d_C,CL_TRUE,0,sizeof(bench_t)*size,h_C, NULL, device_object->evt_copyC);
}

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    device_object->evt_copyC->wait();
    float elapsed_h_d = 0, elapsed = 0, elapsed_d_h = 0;
    elapsed_h_d = device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyA->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    elapsed_h_d += device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyB->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Host->Device: %.10f \n", elapsed / 1000000.0);
    elapsed = device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time kernel: %.10f \n", elapsed / 1000000.0);
    elapsed_d_h = device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_END>() - device_object->evt_copyC->getProfilingInfo<CL_PROFILING_COMMAND_START>();
    //printf("Elapsed time Device->Host: %.10f \n", );


    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", elapsed_h_d / 1000000.0,elapsed / 1000000.0,elapsed_d_h / 1000000.0);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", (elapsed_h_d / 1000000.0));
         printf("Elapsed time kernel: %.10f miliseconds\n", elapsed / 1000000.0);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", elapsed_d_h / 1000000.0);
    }
    return elapsed / 1000000.0; // TODO Change
}

void clean(GraficObject *device_object){
	// pointers clean
    delete device_object->context;
    delete device_object->queue;
    // pointer to memory
    delete device_object->d_A;
    delete device_object->d_B;
    delete device_object->d_C;
    delete device_object->evt;
    delete device_object->evt_copyA;
    delete device_object->evt_copyB;
    delete device_object->evt_copyC;
}
