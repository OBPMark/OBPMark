#htvar kernel_code

/* ATOMIC operations */
//ADD
inline void atomicAdd_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}
//MIN
inline void atomicMin_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = min(expected.f32, val);
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}
//MAX
inline void atomicMax_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = max(expected.f32, val);
        current.u32 = atomic_cmpxchg( (volatile __global unsigned int *)addr, expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
}

/* COMPLEX number support */
typedef float2 cfloat;

inline cfloat cmul(cfloat a, cfloat b){
    return (cfloat) (a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline cfloat conj(cfloat a){
    return (cfloat) (a.x, -a.y);
}

inline cfloat polar1(float th){
    return (cfloat) (cos(th), sin(th));
}

inline float cabs(cfloat a){
    return sqrt(a.x * a.x + a.y * a.y);
}

inline float arg(cfloat a){
    if(a.x > 0){
        return atan(a.y / a.x);

    }else if(a.x < 0 && a.y >= 0){
        return atan(a.y / a.x) + M_PI;

    }else if(a.x < 0 && a.y < 0){
        return atan(a.y / a.x) - M_PI;

    }else if(a.x == 0 && a.y > 0){
        return M_PI/2;

    }else if(a.x == 0 && a.y < 0){
        return -M_PI/2;

    }else{
        return 0;
    }
}


/* OPENCL kernels */

static global float fDc = 0;
static global float v_max = FLT_MIN;
static global float v_min = FLT_MAX;

const float pi = (float) M_PI;      //PI
const float c = (float) 299792458;  //speed of light

inline unsigned int next_power_of2(unsigned int n)
{
    unsigned int v = n;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void kernel
SAR_range_ref(global float *rrf,  const unsigned int rsize, const float fs, const float slope, const unsigned int nit)
{
    cfloat *c_ref = (cfloat*) rrf;
    int i = get_global_id(0);
    if (i >= rsize || i >= nit) return;
    float phase = (-((float)nit)/2+i) * 1/fs;
    phase = pi * slope * phase * phase;
    c_ref[i] = polar1(phase);
}

void kernel
SAR_DCE(global const float *data, const unsigned int apatch, const unsigned int rsize, const float const_k, local cfloat *tmp)
{
    unsigned int i = get_local_id(0);
    if(i >= (apatch-1)) return;
    unsigned int j = get_group_id(0);
    int k = get_num_groups(0);
    unsigned int width = rsize;
    unsigned int off = next_power_of2(width);
    unsigned int cell_x_thread = ceil((float) apatch/get_local_size(0));
    cfloat *c_data = (cfloat*) data;

    tmp[i] = cmul(conj(c_data[i*off+j]), c_data[(i+1)*off+j]);
    for(int k = 1; k < cell_x_thread; k++){
        int l = i + get_local_size(0) * k;
        if (l < apatch-1) tmp[i] += cmul(conj(c_data[l*off+j]), c_data[(l+1)*off+j]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0)/2; s > 0; s >>= 1){
        if (i < s) tmp[i] += tmp[i+s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i == 0){
        float val = arg(tmp[0]);
        val = val*const_k;
        atomicAdd_f(&fDc, val);
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

void kernel
printfDc()
{
    printf("fDc: %f\n", fDc);
}

void kernel
SAR_rcmc_table(global unsigned int *offsets, const unsigned int apatch, const unsigned int avalid, const float PRF, const float lambda, const float vr, const float ro, const float fs)
{
    unsigned int i = get_global_id(1); 
    unsigned int j = get_global_id(0); 

    float delta, offset;
    unsigned int ind;
    unsigned int width = apatch;

    delta = j * (PRF/avalid) + fDc;
    offset = (1/sqrt(1-pow(lambda * delta / (2 * vr), 2))-1) * (ro + i * (c/(2*fs)));
    offset = round (offset / (c/(2*fs))) * width;
    ind = i * width + j;
    offsets[ind] = ind + offset;
}

void kernel
SAR_azimuth_ref(global float *arf, const float ro, const float fs, const float lambda, const float vr, const float PRF, const unsigned int rvalid, const unsigned int apatch)
{
    float  rnge = ro+(rvalid/2)*(c/(2*fs));        //range perpendicular to azimuth
    float  rdc = rnge/sqrt(1-pow(lambda*fDc/(2*vr),2));    //squinted range
    float  tauz = (rdc*(lambda/10) * 0.8) / vr;            //Tau in the azimuth
    float  chirp = -(2*vr*vr)/lambda/rdc;          //Azimuth chirp rate
    int    nit = floor(tauz * PRF);

    cfloat *c_ref = (cfloat*) arf;
    int i = get_global_id(0);
    if (i >= apatch || i >= nit) return;
    float phase = (-((float)nit)/2+i) * 1/PRF;
    phase = 2 * pi * fDc * phase + pi * chirp * phase * phase;
    c_ref[i] = polar1(phase);
}

void kernel
SAR_ref_product(global float *data, global float *ref, const unsigned int w, const unsigned int h)
{
    unsigned int i = get_global_id(1); if (i >= h) return;
    unsigned int j = get_global_id(0); if (j >= w) return;
    i = i + h * get_global_id(2); //Add to i the patch offset
    cfloat *c_data = (cfloat*) data;
    cfloat *c_ref = (cfloat*) ref;
    c_data[i*w+j] = cmul(conj(c_ref[j]), c_data[i*w+j]);
}

void kernel
SAR_transpose(global float *in, global float *out, unsigned int in_width, unsigned int out_width, unsigned int nrows, unsigned int ncols)
{
    unsigned int i = get_global_id(1); if (i >= nrows) return;
    unsigned int j = get_global_id(0); if (j >= ncols) return;
    unsigned int k = get_global_id(2);
    out[2*((j+ncols*k)*out_width+i)]   = in[2*((i + nrows * k) * in_width +j)];
    out[2*((j+ncols*k)*out_width+i)+1] = in[2*((i + nrows * k) * in_width +j)+1];
}

void kernel
SAR_rcmc(global float *data, global unsigned int *offsets, unsigned int width, unsigned int height)
{

    unsigned int i = get_global_id(1);
    unsigned int j = get_global_id(0);
    unsigned int k = get_global_id(2);
    cfloat *c_data = &((cfloat*) data)[(i+k*height) * width];
    unsigned int ind = i * width + j;
    if (offsets[ind] < (height * width)) c_data[ind] = c_data[offsets[ind]];
}

void kernel
SAR_multilook(global float *radar_data, global float *image, const unsigned int rvalid, const unsigned int asize, const unsigned int rsize, const unsigned int npatch, const unsigned int apatch, unsigned int width, unsigned int height)
{
    cfloat *c_data = (cfloat*) radar_data;

    unsigned int isx = rvalid/width; 
    unsigned int isy = asize/width; 
    unsigned int range_w = next_power_of2(rsize);

    int x = get_global_id(1); //if (x >= width) return;
    int y = get_global_id(0); //if (y >= height) return;
    unsigned int oIdx = y * width + x;

    unsigned int row_x_patch = height/npatch;
    unsigned int patch = y / row_x_patch;
    unsigned int patch_offset = patch * apatch * range_w;
    unsigned int initIdx = patch_offset + (y%row_x_patch * isy) * range_w + (x * isx);
    float value;
    float fimg = 0;
    for(int iy = 0; iy < isy; iy++)
        for(int jx = 0; jx < isx; jx++)
            fimg += cabs(c_data[initIdx+iy*range_w+jx]);

    value = fimg/(isx*isy);
    value = (value == 0)?0:log2(value);
    image[oIdx] = value;
    atomicMax_f(&v_max, value);
    barrier(CLK_GLOBAL_MEM_FENCE);
    atomicMin_f(&v_min, value);
    barrier(CLK_GLOBAL_MEM_FENCE);
}

void kernel
quantize(global float *data, global unsigned char *image, const unsigned int width, const unsigned int height)
{
    float scale = 256.f / (v_max-v_min);
    int x = get_global_id(1); if(x >= width) return;
    int y = get_global_id(0); if(y >= height) return;
    image[y*width+x] = min(255.f,floor(scale * (data[y*width+x]-v_min)));
}

//FFT
void kernel
bin_reverse(global float *data, const unsigned int size, const unsigned int group)
{

    unsigned int i = get_global_id(1);
    cfloat *c_data = (cfloat*) &data[i*size*2];
    unsigned int id = get_global_id(0);
    if (id < size)
    {
        unsigned int j = id;
        j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;
        j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;
        j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;
        j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;
        j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;
        j >>= (32-group);
        if(j <= id) return;
        cfloat swp = c_data[id];
        c_data[id] = c_data[j];
        c_data[j] = swp;
    }
}

void kernel
fft_kernel(global float *data, const int loop, const float wpr, const float wpi, unsigned int theads, unsigned int size){
    unsigned int row = get_global_id(1);
    float *B = &data[row*size*2];
    float tempr, tempi;
    unsigned int i = get_global_id(0);
    unsigned int id;
    unsigned int j;
    unsigned int inner_loop;
    unsigned  int subset;
    float wr = 1.0;
    float wi = 0.0;
    float wtemp = 0;

    // get inner
    subset = theads / loop;
    id = i % subset;
    inner_loop = i / subset;
    //get wr and wi
    for(unsigned int z = 0; z < inner_loop ; ++z){
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;

        }
    // get I
    i = id *(loop * 2 * 2) + 1 + (inner_loop * 2);
    j=i+(loop * 2 );

    tempr = wr*B[j-1] - wi*B[j];
    tempi = wr * B[j] + wi*B[j-1];

    B[j-1] = B[i-1] - tempr;
    B[j] = B[i] - tempi;
    B[i-1] += tempr;
    B[i] += tempi;
}


#htendvar
