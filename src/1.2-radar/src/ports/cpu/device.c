/**
 * \file device.c
 * \brief Benchmark #1.2 CPU version (sequential) device initialization. 
 * \author Marc Sole Bonet (BSC)
 */
#include "device.h"
#include "processing.h"

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
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");

}


bool device_memory_init(
	radar_data_t *radar_data,
	radar_params_t *params,
    unsigned int out_height,
    unsigned int out_width
	)
{	
    unsigned int patch_width = params->rsize<<1;
    unsigned int patch_extended_width = next_power_of2(params->rsize)<<1;
    unsigned int patch_height = params->apatch;

	/* radar_data_t memory allocation */
	//RANGE & AZIMUTH DATA
	radar_data->range_data = (framefp_t *) malloc(sizeof(framefp_t) * params->npatch);
	radar_data->azimuth_data = (framefp_t *) malloc(sizeof(framefp_t) * params->npatch);
	for(int i = 0; i < params->npatch; i++)
    {
        radar_data->range_data[i].f = (float*) calloc(patch_height*patch_extended_width, sizeof(float));
        radar_data->range_data[i].h = patch_height;
        radar_data->range_data[i].w = patch_extended_width;

        radar_data->azimuth_data[i].f = (float *) calloc(patch_width * patch_height, sizeof(float));
        radar_data->azimuth_data[i].h = params->rvalid;
        radar_data->azimuth_data[i].w = patch_height<<1;
    }
  	//OUTPUT DATA
  	radar_data->output_data.f = (float*) malloc(sizeof(float)*out_width*out_height);
  	radar_data->output_data.w = out_width;
  	radar_data->output_data.h = out_height;

  	//PARAMS
  	radar_data->params = (radar_params_t*) malloc(sizeof(radar_params_t));
    
    //RANGE REF. FUNCTION
    radar_data->rrf = (float*) calloc(patch_extended_width, sizeof(float));

	//AZIMUTH REF. FUNCTION
    radar_data->arf = (float*) calloc(patch_height<<1, sizeof(float));

    //DOPPLER AUXILIAR BUFFER
    radar_data->aux = (float*) calloc(patch_width, sizeof(float));

    //RCMC TABLE
    radar_data->offsets = (uint32_t *) malloc(sizeof(uint32_t) * params->rvalid * patch_height);

    return true;
}

void copy_memory_to_device(
	radar_data_t *radar_data,
	radar_time_t *t,
	framefp_t *input_data,
	radar_params_t *input_params
	)
{
    /* Copy params */
    memcpy(radar_data->params, input_params, sizeof(radar_params_t));
    uint32_t width = radar_data->params->rsize<<1;
    uint32_t height = radar_data->params->apatch;
    uint32_t line_width = next_power_of2(width);
    for (uint32_t i = 0; i < radar_data->params->npatch; i++ )
        for(uint32_t j = 0; j < height; j++)
            memcpy(&radar_data->range_data[i].f[j * line_width], &input_data[i].f[j * width], width * sizeof(float));
}


void process_benchmark(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{    

	/* Loop through each frames and perform pre-processing. */
	T_START(t->t_test);
    SAR_focus(radar_data);
	T_STOP(t->t_test);

}

void copy_memory_to_host(
	radar_data_t *radar_data,
	radar_time_t *t,
	framefp_t *output_radar
	)
{
    uint32_t  width = radar_data->output_data.w;
    uint32_t  height = radar_data->output_data.h;
    memcpy(output_radar->f, radar_data->output_data.f, sizeof(float) * width * height);
}


void get_elapsed_time(
	radar_data_t *radar_data, 
	radar_time_t *t, 
	bool csv_format,
	bool database_format,
	bool verbose_print,
	long int timestamp
	)
{	

	if (csv_format)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;\n", (float) 0, elapsed_time, (float) 0);
	}
	else if (database_format)
	{
		
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("%.10f;%.10f;%.10f;%ld;\n", (float) 0, elapsed_time, (float) 0, timestamp);
	}
	else if(verbose_print)
	{
		double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000)); 
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (float) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", elapsed_time );
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (float) 0);
	}
}


void clean(
	radar_data_t *radar_data,
	radar_time_t *t
	)
{

	/* Clean time */
	free(t);

	/* Clean radar data */
	for(int i = 0; i < radar_data->params->npatch; i++)
    {
        free(radar_data->range_data[i].f);
        free(radar_data->azimuth_data[i].f);
    }
    free(radar_data->range_data);
    free(radar_data->azimuth_data);
    free(radar_data->output_data.f);
    free(radar_data->params);
    free(radar_data->rrf);
    free(radar_data->arf);
    free(radar_data->offsets);
    free(radar_data->aux);
    free(radar_data);
}
