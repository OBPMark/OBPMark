/** 
 * \brief OBPMark "Radar image generation" processing task.
 * \file processing.c
 * \author david.steenari@esa.int
 * European Space Agency Community License V2.3 applies.
 * For more info see the LICENSE file in the root folder.
 */
#include "processing.h"

void fft(float *data, int nn);
void ifft(float *data, int nn);
void complex_transpose(framefp_t *in, framefp_t *out, uint32_t nrows, uint32_t ncols);

uint32_t next_power_of2(uint32_t n)
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

#ifdef FFT_LIB 
void normalize(float *data, int nn)
{
	for(int i = 0; i<nn; i++){
		data[2*i  ] = data[2*i  ]/nn;
		data[2*i+1] = data[2*i+1]/nn;
	}
}
#endif

void ref_func(float *ref, float fc, float slope, float tau, float fs, uint32_t length, uint32_t fftlen
#ifdef FFT_LIB
		, fftwf_plan plan
#endif 
	     )
	
{
    std::complex<float> *c_ref = (std::complex<float>*) ref;
    uint32_t nit = floor(tau * fs);
    float phase;
    for(int i = 0; i < nit && i < length; i++)
    {
        phase = (-((float)nit)/2+i) * 1/fs;
        phase = 2 * pi * fc * phase + pi * slope * phase * phase;
        c_ref[i] = std::polar(1.f,phase);
    }
#ifdef FFT_LIB
    fftwf_execute(plan);
#else
    fft(ref, fftlen);
#endif
}

void SAR_range_ref(float *rrf, radar_params_t *params
#ifdef FFT_LIB
		, fftwf_plan plan
#endif
		)
{
    ref_func(rrf, 0, params->slope, params->tau, params->fs, params->rsize, next_power_of2(params->rsize)
#ifdef FFT_LIB
		    , plan
#endif 
		    );
}

void SAR_rcmc_table(radar_params_t *params, uint32_t *offsets, float fDc, uint32_t width)
{
    float delta, offset;
    uint32_t ind;

    for (int i = 0; i < params->rvalid; i++)
    {
        for (int j = 0; j < width; j++)
        {
            delta = j * (params->PRF/params->avalid) + fDc;
            offset = (1/sqrt(1-pow(params->lambda * delta / (2 * params->vr), 2))-1) * (params->ro + i * (c/(2*params->fs)));
            offset = round (offset / (c/(2*params->fs)));
            ind = i * width + j;
            offsets[ind] = (int)offset + i;
        }
    }
}

float SAR_DCE(float *aux, framefp_t data, radar_params_t *params)
{
    std::complex<float> *c_data = (std::complex<float>*) data.f;
    std::complex<float> *c_aux = (std::complex<float>*) aux;

    uint32_t width = params->rsize;
    uint32_t off = next_power_of2(width);
    float mean = 0;
    for (int i = 0; i < data.h-1; i++)
        for(int j = 0; j < width; j++){
            c_aux[j] += std::conj(c_data[i*off+j]) * c_data[(i+1)*off+j]; 
            if (i == data.h-2) mean += std::arg(c_aux[j]);
        }
    
    mean = mean/width;
    return mean*params->PRF/(2*pi);
}

void SAR_azimuth_ref(float *arf, radar_params_t *params, float fDc
#ifdef FFT_LIB
		, fftwf_plan plan
#endif 
		)
{
    //Compute parameters for azimuth
    float rnge = params->ro+(params->rvalid/2)*(c/(2*params->fs));        //range perpendicular to azimuth
    float rdc = rnge/sqrt(1-pow(params->lambda*fDc/(2*params->vr),2));    //squinted range
    float tauz = (rdc*(params->lambda/10) * 0.8) / params->vr;            //Tau in the azimuth
    float chirp = -(2*params->vr*params->vr)/params->lambda/rdc;          //Azimuth chirp rate

    ref_func(arf, fDc, chirp, tauz, params->PRF, params->apatch, params->apatch
#ifdef FFT_LIB
		    , plan
#endif 
		    );
}

void reference_multiplication(framefp_t *data, float *ref)
{
    std::complex<float> *c_data = (std::complex<float>*) data->f;
    std::complex<float> *c_ref = (std::complex<float>*) ref;

    uint32_t width = data->w >> 1;
    for (int i = 0; i < data->h; i++)
        for (int j = 0; j < width ; j++)
            c_data[i*width+j] = c_data[i*width+j] * std::conj(c_ref[j]);
}

void SAR_range_compression(framefp_t *data, float *rrf
#ifdef FFT_LIB
		, fftwf_plan forw, fftwf_plan inv
#endif 
		)
{
    //fft data by rows
#ifdef FFT_LIB
    fftwf_execute(forw);
#else
    for(int k = 0; k<data->h; k++) fft(&data->f[k*data->w], data->w>>1);
#endif

    //multply with conjugate of reference function
    reference_multiplication(data, rrf);

    //return to freq domain with ifft
#ifdef FFT_LIB
    fftwf_execute(inv);
    for(int k = 0; k<data->h; k++) normalize(&data->f[k*data->w], data->w>>1);
#else
    for(int k = 0; k<data->h; k++) ifft(&data->f[k*data->w], data->w>>1);
#endif
    
}

void SAR_rcmc(framefp_t *data, uint32_t *offsets
#ifdef FFT_LIB
		, fftwf_plan plan
#endif
	     )
{
    std::complex<float> *c_data = (std::complex<float>*) data->f;

    //fft data by rows
#ifdef FFT_LIB
    fftwf_execute(plan);
#else
    for(int k = 0; k<data->h; k++) fft(&data->f[k*data->w], data->w>>1);
#endif

    //RCMC
    uint32_t height = data->h;
    uint32_t width = data->w>>1;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            uint32_t ind = i * width + j;
            if(offsets[ind]<height) c_data[ind] = c_data[j+offsets[ind]*width];
            else c_data[ind] = 0;
        }
}

void SAR_azimuth_compression(framefp_t *data, float *arf
#ifdef FFT_LIB
		, fftwf_plan plan
#endif
		)
{
    //multiply row by arf
    reference_multiplication(data,arf);

    //return to freq domain with ifft
#ifdef FFT_LIB
    fftwf_execute(plan);
    for(int k = 0; k<data->h; k++) normalize(&data->f[k*data->w], data->w>>1);
#else
    for(int k = 0; k<data->h; k++) ifft(&data->f[k*data->w], data->w>>1);
#endif
}

void SAR_multilook(framefp_t *radar_data, framefp_t *image, radar_params_t *params, uint32_t patch, float *max, float *min)
{
    std::complex<float> *c_data = (std::complex<float>*) radar_data->f;

    float sx = (float)params->asize/(float)image->w;
    float sy = (float)params->rvalid/(float)image->w;
    uint32_t nfx = floor(params->avalid/sx);
    uint32_t nfy = floor(params->rvalid/sy);

    uint32_t isx = floor(sx);
    uint32_t isy = floor(sy);
    float value;
    uint32_t offset = patch * nfx * nfy;
    float fimg;
    for(int i = 0; i < nfx; i++)
        for(int j = 0; j < nfy; j++){
            fimg = 0;
            for(int ix = 0; ix < isx; ix++)
                for(int jy = 0; jy < isy; jy++)
                    fimg += std::abs(c_data[(i*isx+ix)*(next_power_of2(params->rsize))+j*isy+jy]);
            value = fimg/(isx*isy);
            value = (value == 0)?0:log2(value);
            image->f[offset+i*image->w+j] = value;
            *max = (*max<value)?value:*max;
            *min = (*min>value)?value:*min;
        }
}

void quantize(framefp_t *data, frame8_t *image, float max, float min)
{
    float scale = 256.f / (max-min);
    for (int i = 0; i < image->h; i++){
        for (int j = 0; j < image->w; j++){
            image->f[i*image->w+j] = std::min(255.f,floor(scale * (data->f[i*image->w+j]-min)));
        }
    }
}

void SAR_focus(radar_data_t *data){
    float max = FLT_MIN;
    float min = FLT_MAX;
    /* Compute Range Reference Function */
    SAR_range_ref(data->rrf, data->params
#ifdef FFT_LIB
		    , data->rrf_plan
#endif
		    );

    /* Compute Doppler Centroid */
    float fDc = SAR_DCE(data->aux, data->range_data[0], data->params);

    /* Create RCMC table */
    SAR_rcmc_table(data->params, data->offsets, fDc, data->params->apatch);

    /* Compute Azimuth Reference Function */
    SAR_azimuth_ref(data->arf, data->params, fDc
#ifdef FFT_LIB
		    , data->arf_plan
#endif
		    );

    /* Begin Patch computation loop */
#pragma omp parallel for
    for (int i = 0; i < data->params->npatch; i++)
    {
        /* Range Compression */
        SAR_range_compression(&data->range_data[i], data->rrf
#ifdef FFT_LIB
			, data->range_plan[i], data->range_plan_inv[i]
#endif 
			);

        /* Transpose to operate with Azimuth data */
        complex_transpose(&data->range_data[i], &data->azimuth_data[i], data->params->apatch, data->params->rvalid);

        /* Range Cell Migration Correction */
        SAR_rcmc(&data->azimuth_data[i], data->offsets
#ifdef FFT_LIB
			, data->azimuth_plan[i]
#endif
			);

        /* Azimuth Compression */
        SAR_azimuth_compression(&data->azimuth_data[i], data->arf
#ifdef FFT_LIB
			, data->azimuth_plan_inv[i]
#endif 
			);

        /* Transpose back to Range data */
        complex_transpose(&data->azimuth_data[i], &data->range_data[i], data->params->rvalid, data->params->apatch);

        /* Multilook */
        SAR_multilook(&data->range_data[i], &data->ml_data, data->params, i, &max, &min);
    }
    quantize(&data->ml_data, &data->output_image, max, min);
}

//From GPU4S_benchmarks
void fft_kernel(float* data, int nn, int isign){
	int n, mmax, m, j, istep, i;
    float wtemp, wr, wpr, wpi, wi, theta;
    float tempr, tempi;
 
    // reverse-binary reindexing
    n = nn<<1;
    j=1;
    for (i=1; i<n; i+=2) {
        if (j>i) {
            std::swap(data[j-1], data[i-1]);
            std::swap(data[j], data[i]);
        }
        m = nn;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    };

    // here begins the Danielson-Lanczos section
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*(float)M_PI/(mmax*isign));
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*data[j-1] - wi*data[j];
                tempi = wr * data[j] + wi*data[j-1];
 				
                data[j-1] = data[i-1] - tempr;
                data[j] = data[i] - tempi;
                data[i-1] += tempr;
                data[i] += tempi;
            }
            
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;

        }
        mmax=istep;
    }
}

void fft(float* data, int nn){
    fft_kernel(data, nn, 1);
}

void ifft(float* data, int nn){
    fft_kernel(data, nn, -1);
    for(int i = 0; i<nn; i++){
        data[2*i  ] = data[2*i  ]/nn;
        data[2*i+1] = data[2*i+1]/nn;
    }
}

void complex_transpose(framefp_t *in, framefp_t *out, uint32_t nrows, uint32_t ncols){
    std::complex<float> *c_a = (std::complex<float>*) in->f;
    std::complex<float> *c_b = (std::complex<float>*) out->f;
    unsigned int a_rw = in->w / 2;
    unsigned int b_rw = out->w / 2;
    for(int i = 0; i<nrows; i++)
        for(int j = 0; j<ncols; j++)
            c_b[j*b_rw+i] = c_a[i*a_rw+j];
}
