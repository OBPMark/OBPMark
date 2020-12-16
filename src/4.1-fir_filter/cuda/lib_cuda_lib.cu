#include <cudnn.h>
#include "../benchmark_library.h"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#ifdef INT
  #define CUDNNTYPE CUDNN_DATA_INT32
#elif FLOAT
  #define CUDNNTYPE CUDNN_DATA_FLOAT
#elif DOUBLE
  #define CUDNNTYPE CUDNN_DATA_DOUBLE
#endif

void init(GraficObject *device_object, char* device_name){
    init(device_object, 0,0, device_name);
}

void init(GraficObject *device_object, int platform ,int device, char* device_name){
    cudaSetDevice(device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    //printf("Using device: %s\n", prop.name);
    strcpy(device_name,prop.name);
    //event create 
    device_object->start = new cudaEvent_t;
    device_object->stop = new cudaEvent_t;
    device_object->start_memory_copy_device = new cudaEvent_t;
    device_object->stop_memory_copy_device = new cudaEvent_t;
    device_object->start_memory_copy_host = new cudaEvent_t;
    device_object->stop_memory_copy_host= new cudaEvent_t;
    
    cudaEventCreate(device_object->start);
    cudaEventCreate(device_object->stop);
    cudaEventCreate(device_object->start_memory_copy_device);
    cudaEventCreate(device_object->stop_memory_copy_device);
    cudaEventCreate(device_object->start_memory_copy_host);
    cudaEventCreate(device_object->stop_memory_copy_host);
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix){

   // Allocate the device input vector A
	cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&device_object->d_A, size_a_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device input vector B
    err = cudaMalloc((void **)&device_object->d_B, size_b_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }

    // Allocate the device output vector C
    err = cudaMalloc((void **)&device_object->kernel, size_c_matrix * sizeof(bench_t));

    if (err != cudaSuccess)
    {
        return false;
    }
    return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b){
    cudaEventRecord(*device_object->start_memory_copy_device);
	cudaError_t err = cudaMemcpy(device_object->d_A, h_A, sizeof(bench_t) * size_a, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaMemcpy(device_object->kernel, kernel, sizeof(bench_t) * size_b, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    cudaEventRecord(*device_object->stop_memory_copy_device);   
}
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size){
    // cublas settings
    const bench_t alf = 1;
    const bench_t bet = 0;
    cudnnHandle_t cudnn;

    cudaEventRecord(*device_object->start);
    checkCUDNN(cudnnCreate(&cudnn));
    // create input tensor
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/n));
    // create output tensor
    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                      /*format=*/CUDNN_TENSOR_NHWC,
                                      /*dataType=*/CUDNNTYPE,
                                      /*batch_size=*/1,
                                      /*channels=*/1,
                                      /*image_height=*/1,
                                      /*image_width=*/n));
    // create kernel tensor
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                      /*dataType=*/CUDNNTYPE,
                                      /*format=*/CUDNN_TENSOR_NCHW,
                                      /*out_channels=*/1,
                                      /*in_channels=*/1,
                                      /*kernel_height=*/1,
                                      /*kernel_width=*/kernel_size));
    // create kernel descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                           /*pad_height=*/1,
                                           /*pad_width=*/1,
                                           /*vertical_stride=*/1,
                                           /*horizontal_stride=*/1,
                                           /*dilation_height=*/1,
                                           /*dilation_width=*/1,
                                           /*mode=*/CUDNN_CROSS_CORRELATION,
                                           /*computeType=*/CUDNNTYPE));
    //use tensorcore
    //cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH)
    // describing convolution
    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm));
    // get memory needed for the convolution
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                   input_descriptor,
                                                   kernel_descriptor,
                                                   convolution_descriptor,
                                                   output_descriptor,
                                                   convolution_algorithm,
                                                   &workspace_bytes));
    // alocate memory for workspace
    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);
    // perform the convolution
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                   &alf,
                                   input_descriptor,
                                   device_object->d_A,
                                   kernel_descriptor,
                                   device_object->kernel,
                                   convolution_descriptor,
                                   convolution_algorithm,
                                   d_workspace,
                                   workspace_bytes,
                                   &bet,
                                   output_descriptor,
                                   device_object->d_B));
    
   
    
    cudaEventRecord(*device_object->stop);
    // destroy cuDNN
    cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}

void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size){
    cudaEventRecord(*device_object->start_memory_copy_host);
    cudaMemcpy(h_C, device_object->d_B, size * sizeof(bench_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(*device_object->stop_memory_copy_host);
    }

float get_elapsed_time(GraficObject *device_object, bool csv_format){
    cudaEventSynchronize(*device_object->stop_memory_copy_host);
    float milliseconds_h_d = 0, milliseconds = 0, milliseconds_d_h = 0;
    // memory transfer time host-device
    cudaEventElapsedTime(&milliseconds_h_d, *device_object->start_memory_copy_device, *device_object->stop_memory_copy_device);
    // kernel time
    cudaEventElapsedTime(&milliseconds, *device_object->start, *device_object->stop);
    //  memory transfer time device-host
    cudaEventElapsedTime(&milliseconds_d_h, *device_object->start_memory_copy_host, *device_object->stop_memory_copy_host);
    
    if (csv_format){
         printf("%.10f;%.10f;%.10f;\n", milliseconds_h_d,milliseconds,milliseconds_d_h);
    }else{
         printf("Elapsed time Host->Device: %.10f miliseconds\n", milliseconds_h_d);
         printf("Elapsed time kernel: %.10f miliseconds\n", milliseconds);
         printf("Elapsed time Device->Host: %.10f miliseconds\n", milliseconds_d_h);
    }
    return milliseconds;
}

void clean(GraficObject *device_object){
	cudaError_t err = cudaSuccess;
	err = cudaFree(device_object->d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }

    err = cudaFree(device_object->d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        return;
    }
    err = cudaFree(device_object->kernel);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        return;
    }



    // delete events
    delete device_object->start;
    delete device_object->stop;
    delete device_object->start_memory_copy_device;
    delete device_object->stop_memory_copy_device;
    delete device_object->start_memory_copy_host;
    delete device_object->stop_memory_copy_host;
}