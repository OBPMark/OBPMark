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

void print_data(framefp_t *data, int npatch)
{
	char* output_file = (char*)"debug.txt";
	FILE *framefile;
	framefile = fopen(output_file, "w");
	if(framefile == NULL) {
		printf("error: failed to open file: %s\n", output_file);
	}
	unsigned int h_position; 
	unsigned int w_position;
    for(int i = 0; i < npatch; i++){
        fprintf(framefile,"Patch %d/%d:\n", i+1, npatch);
        std::complex<float> *c_data = (std::complex<float>*) data[i].f;

        /* Print output */
        for(h_position=0; h_position < data[i].h; h_position++)
        {
            for(w_position=0; w_position < data[i].w/2; w_position++)
            {
                std::complex<float> val = c_data[(h_position * (data[i].w/2) + w_position)];
                fprintf(framefile, "% 20.10f", real(val));
                fprintf(framefile, "%+20.10fi ", imag(val));

            }
            fprintf(framefile,"\n");
        }
    }
}

void print_output(framefp_t *output_image)
{
	unsigned int h_position; 
	unsigned int w_position;

	/* Print output */
	for(h_position=0; h_position < output_image->h; h_position++)
	{
		
		for(w_position=0; w_position < output_image->w; w_position++)
		{
			//FIXME chaneg to the 1D and 2D version
			printf("%f, ", output_image->f[(h_position * (output_image->w) + w_position)]);
		}
		printf("\n");
	}
}

void print_params(radar_params_t *params)
{
    printf("Lambda: %.24f\n", params->lambda);
    printf("PRF: %.24f\n", params->PRF);
    printf("Tau: %.24f\n", params->tau);
    printf("Fs: %.24f\n", params->fs);
    printf("Vr: %.24f\n", params->vr);
    printf("Ro: %.24f\n", params->ro);
    printf("Slope: %.24f\n", params->slope);
    printf("Asize: %d\n", params->asize);
    printf("Avalid: %d\n", params->avalid);
    printf("Apatch: %d\n", params->apatch);
    printf("Rsize: %d\n", params->rsize);
    printf("NPatch: %d\n", params->npatch);
}

void ref_func(float *ref, float fc, float slope, float tau, float fs, uint32_t length, uint32_t fftlen)
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
    fft(ref, fftlen);
}

void SAR_range_ref(float *rrf, radar_params_t *params)
{
    ref_func(rrf, 0, params->slope, params->tau, params->fs, params->rsize, next_power_of2(params->rsize));

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
            offset = round (offset / (c/(2*params->fs))) * width;
            ind = i * width + j;
            offsets[ind] = ind + offset;
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

void SAR_azimuth_ref(float *arf, radar_params_t *params, float fDc)
{
    //Compute parameters for azimuth
    float rnge = params->ro+(params->rvalid/2)*(c/(2*params->fs));        //range perpendicular to azimuth
    float rdc = rnge/sqrt(1-pow(params->lambda*fDc/(2*params->vr),2));    //squinted range
    float tauz = (rdc*(params->lambda/10) * 0.8) / params->vr;            //Tau in the azimuth
    float chirp = -(2*params->vr*params->vr)/params->lambda/rdc;          //Azimuth chirp rate

    ref_func(arf, fDc, chirp, tauz, params->PRF, params->apatch, params->apatch);
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

void SAR_range_compression(framefp_t *data, float *rrf)
{
    //fft data by rows
    for(int k = 0; k<data->h; k++)
        fft(&data->f[k*data->w], data->w>>1);

    //multply with conjugate of reference function
    reference_multiplication(data, rrf);

    //return to freq domain with ifft
    for(int k = 0; k<data->h; k++)
        ifft(&data->f[k*data->w], data->w>>1);
    
}

void SAR_rcmc(framefp_t *data, uint32_t *offsets)
{
    std::complex<float> *c_data = (std::complex<float>*) data->f;

    //fft data by rows
    for(int k = 0; k<data->h; k++)
        fft(&data->f[k*data->w], data->w>>1);

    //RCMC
    uint32_t height = data->h;
    uint32_t width = data->w>>1;

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            uint32_t ind = i * width + j;
            c_data[ind] = (offsets[ind]<(height*width)?c_data[offsets[ind]]:0);
        }
}

void SAR_azimuth_compression(framefp_t *data, float *arf)
{
    //multiply row by arf
    reference_multiplication(data,arf);

    //return to freq domain with ifft
    for(int k = 0; k<data->h; k++)
        ifft(&data->f[k*data->w], data->w>>1);
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
    //    print_params(data->params);
    SAR_range_ref(data->rrf, data->params);

    /* Compute Doppler Centroid */
    float fDc = SAR_DCE(data->aux, data->range_data[0], data->params);

    /* Create RCMC table */
    SAR_rcmc_table(data->params, data->offsets, fDc, data->params->apatch);

    /* Compute Azimuth Reference Function */
    SAR_azimuth_ref(data->arf, data->params, fDc);

    /* Begin Patch computation loop */
#pragma omp parallel for
    for (int i = 0; i < data->params->npatch; i++)
    {
        /* Range Compression */
        SAR_range_compression(&data->range_data[i], data->rrf);

        /* Transpose to operate with Azimuth data */
        complex_transpose(&data->range_data[i], &data->azimuth_data[i], data->params->apatch, data->params->rvalid);

        /* Range Cell Migration Correction */
        SAR_rcmc(&data->azimuth_data[i], data->offsets);

        /* Azimuth Compression */
        SAR_azimuth_compression(&data->azimuth_data[i], data->arf);

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
