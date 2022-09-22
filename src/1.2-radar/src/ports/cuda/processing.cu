/** 
 * \brief OBPMark "Image corrections and calibrations." processing task.
 * \file processing.c
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "processing.h"

__device__ void fft(float *data, int nn);
__device__ void ifft(float *data, int nn);
__device__ void complex_transpose(float *in, float *out, uint32_t nrows, uint32_t ncols, uint32_t in_width, uint32_t out_width);

__device__ uint32_t next_power_of2(uint32_t n)
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


static __device__ float fDc = 0;
static __device__ float v_max = FLT_MIN;
static __device__ float v_min = FLT_MAX;
/* Debug kernels */
__global__ void printffDc()
{
    printf("fDc: %.12f\n",fDc);
}
__global__ void printfM()
{
    printf("Max: %.12f\n",v_max);
    printf("Min: %.12f\n",v_min);
}


__global__ void SAR_range_ref(float *rrf, radar_params_t *params, uint32_t nit)
{
    cuda::std::complex<float> *c_ref = (cuda::std::complex<float>*) rrf;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= params->rsize || i >= nit) return;
    float phase = (-((float)nit)/2+i) * 1/params->fs;
    phase = pi * params->slope * phase * phase;
    c_ref[i] = cuda::std::polar(1.f,phase);
}


__global__ void SAR_rcmc_table(radar_params_t *params, uint32_t *offsets)
{
    uint32_t i = blockIdx.y * TILE_SIZE + threadIdx.y; if (i >= params->rvalid) return;
    uint32_t j = blockIdx.x * TILE_SIZE + threadIdx.x; if (j >= params->apatch) return;

    float delta, offset;
    uint32_t ind;
    uint32_t width = params->apatch;

    delta = j * (params->PRF/params->avalid) + fDc;
    offset = (1/sqrt(1-pow(params->lambda * delta / (2 * params->vr), 2))-1) * (params->ro + i * (c/(2*params->fs)));
    offset = round (offset / (c/(2*params->fs))) * width;
    ind = i * width + j;
    offsets[ind] = ind + offset;
}

__global__ void SAR_DCE(float *data, radar_params_t *params, float const_k)
{
    uint32_t i = threadIdx.x;
    if(i >= (params->apatch-1)) return;
    uint32_t j = blockIdx.x;
    uint32_t width = params->rsize;
    uint32_t off = next_power_of2(width);
    uint32_t cell_x_thread = ceil((float) params->apatch/BLOCK_SIZE);
    cuda::std::complex<float> *c_data = (cuda::std::complex<float>*) data;
    extern __shared__ cuda::std::complex<float> tmp[];

    tmp[i] = cuda::std::conj(c_data[i*off+j]) * c_data[(i+1)*off+j];
    for(int k = 1; k < cell_x_thread; k++){
        int l = i + BLOCK_SIZE * k;
        if (l < params->apatch-1) tmp[i] += cuda::std::conj(c_data[l*off+j]) * c_data[(l+1)*off+j];
    }
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1){
        if (i < s) tmp[i] += tmp[i+s];
        __syncthreads();
    }
    if (i == 0){
        float val = cuda::std::arg(tmp[0]);
        val = val*const_k;
        __threadfence();
        atomicAdd(&fDc, val);
    }
}

__global__ void SAR_azimuth_ref(float *arf, radar_params_t *params)
{
    float  rnge = params->ro+(params->rvalid/2)*(c/(2*params->fs));        //range perpendicular to azimuth
    float  rdc = rnge/sqrt(1-pow(params->lambda*fDc/(2*params->vr),2));    //squinted range
    float  tauz = (rdc*(params->lambda/10) * 0.8) / params->vr;            //Tau in the azimuth
    float  chirp = -(2*params->vr*params->vr)/params->lambda/rdc;          //Azimuth chirp rate
    int    nit = floor(tauz * params->PRF);

    cuda::std::complex<float> *c_ref = (cuda::std::complex<float>*) arf;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= params->apatch || i >= nit) return;
    float phase = (-((float)nit)/2+i) * 1/params->PRF;
    phase = 2 * pi * fDc * phase + pi * chirp * phase * phase;
    c_ref[i] = cuda::std::polar(1.f,phase);
}

__global__ void SAR_ref_product(float *data, float *ref, uint32_t w, uint32_t h)
{
    uint32_t i = blockIdx.y * TILE_SIZE + threadIdx.y; if (i >= h) return;
    uint32_t j = blockIdx.x * TILE_SIZE + threadIdx.x; if (j >= w) return;
    i = i + h * blockIdx.z; //Add to i the patch offset
    cuda::std::complex<float> *c_data = (cuda::std::complex<float>*) data;
    cuda::std::complex<float>* c_ref = (cuda::std::complex<float>*) ref;
    c_data[i*w+j] *= cuda::std::conj(c_ref[j]);
}

__global__ void SAR_transpose(float *in, float *out, uint32_t in_width, uint32_t out_width, uint32_t nrows, uint32_t ncols)
{
    uint32_t i = blockIdx.y * TILE_SIZE + threadIdx.y; if (i >= nrows) return;
    uint32_t j = blockIdx.x * TILE_SIZE + threadIdx.x; if (j >= ncols) return;
    out[2*((j+ncols*blockIdx.z)*out_width+i)]   = in[2*((i + nrows * blockIdx.z) * in_width +j)]/in_width;
    out[2*((j+ncols*blockIdx.z)*out_width+i)+1] = in[2*((i + nrows * blockIdx.z) * in_width +j)+1]/in_width;
}

__global__ void SAR_rcmc(float *data, uint32_t *offsets, uint32_t width, uint32_t height)
{

    uint32_t i = blockIdx.y * TILE_SIZE + threadIdx.y; if (i >= height) return;
    uint32_t j = blockIdx.x * TILE_SIZE + threadIdx.x; if (j >= width) return;
    cuda::std::complex<float> *c_data = &((cuda::std::complex<float>*) data)[(i+blockIdx.z*height) * width];
    uint32_t ind = i * width + j;
    if (offsets[ind] < (height * width)) c_data[ind] = c_data[offsets[ind]];
}

/* UTIL float atomic Max and atomic Min */
__device__ float atomicMax(float* address, float val)
{
    unsigned int* address_as_uint =(unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    while(val > __uint_as_float(old) ) {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val));
    }

    return __int_as_float(old);
}
__device__ float atomicMin(float* address, float val)
{
    unsigned int* address_as_uint =(unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    while(val < __uint_as_float(old) ) {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val));
    }

    return __uint_as_float(old);
}


__global__ void SAR_multilook(float *radar_data, float *image, radar_params_t *params, uint32_t width, uint32_t height)
{
    cuda::std::complex<float> *c_data = (cuda::std::complex<float>*) radar_data;

    uint32_t isx = params->rvalid/width; 
    uint32_t isy = params->asize/width; 
    uint32_t range_w = next_power_of2(params->rsize);

    int x = blockIdx.x * TILE_SIZE + threadIdx.x; if (x >= width) return;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; if (y >= height) return;
    uint32_t oIdx = y * width + x;

    uint32_t row_x_patch = height/params->npatch;
    uint32_t patch = y / row_x_patch;
    uint32_t patch_offset = patch * params->apatch * range_w;
    uint32_t initIdx = patch_offset + (y%row_x_patch * isy) * range_w + (x * isx);
    float value;
    float fimg = 0;
    for(int iy = 0; iy < isy; iy++)
        for(int jx = 0; jx < isx; jx++)
            fimg += cuda::std::abs(c_data[initIdx+iy*range_w+jx]);

    value = fimg/(isx*isy);
    value = (value == 0)?0:log2(value);
    image[oIdx] = value;
    __threadfence();
    atomicMax(&v_max, value);
    __threadfence();
    atomicMin(&v_min, value);
}

__global__ void quantize(float *data, uint8_t *image, uint32_t width, uint32_t height)
{
    float scale = 256.f / (v_max-v_min);
    int x = blockIdx.x * TILE_SIZE + threadIdx.x; if(x >= width) return;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; if(y >= height) return;
    image[y*width+x] = min(255.f,floor(scale * (data[y*width+x]-v_min)));
}
