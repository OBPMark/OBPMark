/**
 * \file device.c
 * \brief Benchmark #121 CPU version (sequential) device initialization. 
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
#include "device.h"
#include "processing.h"

void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	char *device_name
	)
{
    init(compression_data,t, 0,0, device_name);
}



void init(
	compression_data_t *compression_data,
	compression_time_t *t,
	int platform,
	int device,
	char *device_name
	)
{
    // TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");

}



bool device_memory_init(
	compression_data_t *compression_data
	)
{	
	// do a calloc  OutputPreprocessedValue with size totalSamples
	compression_data->OutputPreprocessedValue = (unsigned int *) calloc(compression_data->TotalSamplesStep, sizeof(unsigned int));
	compression_data->size = (unsigned int *) calloc(compression_data->r_samplesInterval * compression_data->steps, sizeof(unsigned int));
	compression_data->data = (unsigned int *) calloc(compression_data->r_samplesInterval * compression_data->j_blocksize * compression_data->steps, sizeof(unsigned int));
	compression_data->CompressionIdentifier = (unsigned char *) calloc(compression_data->r_samplesInterval * compression_data->steps, sizeof(unsigned char));
	compression_data->CompressionIdentifierInternal = (unsigned int *) calloc(compression_data->r_samplesInterval * compression_data->steps, sizeof(unsigned int));
	return true;
}



void copy_memory_to_device(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
// EMPTY
}


void process_benchmark(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{

	T_START(t->t_test);
	unsigned int step;
	//create a arrat of ZeroBlockCounter with size r_samplesInterval
	//struct ZeroBlockCounter *ZeroCounter = (ZeroBlockCounter *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockCounter));
	//struct ZeroBlockProcessed *ZBProcessed= (ZeroBlockProcessed *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockProcessed));

	
	//#pragma omp parallel for private(step)
	for (step = 0; step < compression_data->steps; ++step)
	{
		// init preprocessed value
		compression_data->DelayedStack = (int *)calloc(1, sizeof(int));

		struct ZeroBlockCounter *ZeroCounter = (ZeroBlockCounter *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockCounter));
		struct ZeroBlockProcessed *ZBProcessed= (ZeroBlockProcessed *) calloc(compression_data->r_samplesInterval, sizeof(ZeroBlockProcessed));
		unsigned int ZeroCounterPos = 0;
		// for each step init zero counter
		for(int i = 0; i < compression_data->r_samplesInterval; ++i) { ZeroCounter[i].counter = 0; ZeroCounter[i].position = -1; }
		preprocess_data(compression_data,&ZeroCounterPos,ZeroCounter, step);
		// ZeroBlock processed array per position
		for(int i = 0; i < compression_data->r_samplesInterval; ++i) { ZBProcessed[i].NumberOfZeros = -1; }
		process_zeroblock(compression_data,&ZeroCounterPos,ZeroCounter,ZBProcessed);
		// Compressing each block
		process_blocks(compression_data, ZBProcessed, step);
		// Free memory
		free(compression_data->DelayedStack);
		free(ZeroCounter);
		free(ZBProcessed);
	}

	// compress the data
	
	for (step = 0; step < compression_data->steps; ++step)
	{
		if(compression_data->debug_mode){printf("Step %d\n",step);}
		for(unsigned int block = 0; block < compression_data->r_samplesInterval; ++block)
    	{
			if(compression_data->debug_mode){printf("Block %d\n",block);}
			// get the CompressionIdentifierInternal
			unsigned int CompressionIdentifierInternal = compression_data->CompressionIdentifierInternal[block + step * compression_data->r_samplesInterval];
			// get the CompressionIdentifier
			unsigned char CompressionIdentifier = compression_data->CompressionIdentifier[block + step * compression_data->r_samplesInterval];
			// get the size
			unsigned int size = compression_data->size[block + step * compression_data->r_samplesInterval];
			// get the offset data pointer
			unsigned int *data = compression_data->data + block * compression_data->j_blocksize + step * compression_data->r_samplesInterval * compression_data->j_blocksize;
			// now prepare to write the compressed data
			unsigned int compression_technique_identifier_size = 1;

			// define the size of the compression technique identifier base of n_bits size
			if (compression_data->n_bits < 3){
				compression_technique_identifier_size = 1;
			}
			else if (compression_data->n_bits < 5)
			{
				compression_technique_identifier_size = 2;
			}
			else if (compression_data->n_bits <= 8)
			{
				compression_technique_identifier_size = 3;
			}
			else if (compression_data->n_bits <= 16)
			{
				compression_technique_identifier_size = 4;
			}
			else 
			{
				compression_technique_identifier_size = 5;
			}

			// If the selected technique is Zero Block or the Second Extension, the compression_technique_identifier_size is +1
			if(CompressionIdentifierInternal == ZERO_BLOCK_ID || CompressionIdentifierInternal == SECOND_EXTENSION_ID)
			{
				compression_technique_identifier_size += 1;
			}
			// The sizeof an unsigned short is 16 bit. We simply can use this conversion method to have a good bit stream for this field.
			const unsigned char CompressionTechniqueIdentifier = CompressionIdentifier;
			writeWordChar(compression_data->OutputDataBlock, CompressionTechniqueIdentifier, compression_technique_identifier_size);

			if(compression_data->debug_mode){
				printf("CompressionTechniqueIdentifier and Size :%u, %u\n",compression_technique_identifier_size, CompressionTechniqueIdentifier);
				unsigned int number_of_elements = 0;
				if (CompressionIdentifierInternal == ZERO_BLOCK_ID)
				{
					for (int i = 0; i < compression_data->j_blocksize; ++i)
					{
						printf("%d ", 0);
					}
					printf("\n");
				}
				else
				{
					if (CompressionIdentifierInternal == SECOND_EXTENSION_ID)
					{
						number_of_elements = compression_data->j_blocksize/2;
					}
					else
					{
						
						number_of_elements = compression_data->j_blocksize;
						
					}
					for (int i = 0; i < number_of_elements; ++i)
					{
						printf("%d ", data[i]);
					}
					printf("\n");
				}
			}

			// Write the compressed blocks based on the selected compression algorithm
			if (CompressionIdentifierInternal == ZERO_BLOCK_ID)
			{
				if(compression_data->debug_mode){printf("Zero block with size %d\n",size);}
				ZeroBlockWriter(compression_data->OutputDataBlock, size);
			}
			else if(CompressionIdentifierInternal == NO_COMPRESSION_ID)
			{
				if(compression_data->debug_mode){printf("No compression with size %d\n",size);}
				NoCompressionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, compression_data->n_bits, data);
			}
			else if(CompressionIdentifierInternal == FUNDAMENTAL_SEQUENCE_ID)
			{
				if(compression_data->debug_mode){printf("Fundamental sequence with size %d\n",size);}
				FundamentalSequenceWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, data);
			}
			else if(CompressionIdentifierInternal == SECOND_EXTENSION_ID)
			{
				if(compression_data->debug_mode){printf("Second extension with size %d\n",size);}
				SecondExtensionWriter(compression_data->OutputDataBlock, compression_data->j_blocksize/2,data);
			}
			else if(CompressionIdentifierInternal >= SAMPLE_SPLITTING_ID)
			{
				if(compression_data->debug_mode){printf("Sample splitting with K %d and size %d\n",CompressionIdentifierInternal - SAMPLE_SPLITTING_ID, size);}
				SampleSplittingWriter(compression_data->OutputDataBlock, compression_data->j_blocksize, CompressionIdentifierInternal - SAMPLE_SPLITTING_ID, data);
			}
			else
			{
				printf("Error: Unknown compression technique identifier\n");
			}
				}
	}


	T_STOP(t->t_test);
	// free ZeroCounter
	

}



void copy_memory_to_host(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
//EMPTY
}


void get_elapsed_time(
	compression_data_t *compression_data, 
	compression_time_t *t, 
	print_info_data_t *benchmark_info,
	long int timestamp
	)
{	
	double elapsed_time =   (t->t_test) / ((double)(CLOCKS_PER_SEC / 1000));
	print_execution_info(benchmark_info, false, timestamp,0,(float)(elapsed_time),0);
}


void clean(
	compression_data_t *compression_data,
	compression_time_t *t
	)
{
	// free all memory from compression_data
	free(compression_data->OutputPreprocessedValue);
	free(compression_data->CompressionIdentifier);
	free(compression_data->CompressionIdentifierInternal);
	free(compression_data->size);
	free(compression_data->data);



}