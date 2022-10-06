#include "lib_functions.h"

#define OK_ARGUMENTS 0
#define ERROR_ARGUMENTS -1
#define NUMBER_BASE 255

int arguments_handler(int argc, char ** argv,unsigned int *w_size, unsigned int *h_size, bool *encode ,bool *generate, bool *type, char *input_image, char *output_image);
void  writeBMP(char* filename, int* output ,unsigned int w_size, unsigned int h_size);
void writeBMP(char* filename, float* output ,unsigned int w_size, unsigned int h_size);
int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns);

int main(int argc, char **argv)
{
unsigned int w_size = 0, h_size = 0;
bool encode = false; // If encode is to false we will perform a decode. If encode is true we will perform a encode
bool type = false; // If type is false we will perform the encode/decode with integer else will perform with float
bool generate = false; // if the image has to be generated
char input_image[100] = "";
char output_image[100] = "";
// process arguments 
int resolution = arguments_handler(argc,argv,&w_size, &h_size,&encode, &type, &generate, input_image, output_image);
if (resolution == ERROR_ARGUMENTS){
		exit(-1);
}
// random seed
srand (21121993);
// base object init
DataObject *ccdsd_data = (DataObject *)malloc(sizeof(DataObject));
// preload data
ccdsd_data->w_size = w_size;
ccdsd_data->h_size = h_size;
ccdsd_data->encode = encode;
ccdsd_data->type = type;
ccdsd_data->filename_input = input_image;
ccdsd_data->filename_output = output_image;
unsigned int pad_rows = 0;
unsigned int pad_columns = 0;


if(ccdsd_data->h_size % BLOCKSIZEIMAGE != 0){
		pad_rows =  BLOCKSIZEIMAGE - (ccdsd_data->h_size % BLOCKSIZEIMAGE);
}
	
if(ccdsd_data->w_size % BLOCKSIZEIMAGE != 0){
    pad_columns = BLOCKSIZEIMAGE - (ccdsd_data->w_size % BLOCKSIZEIMAGE);
}
ccdsd_data->pad_rows = pad_rows;
ccdsd_data->pad_columns = pad_columns;
// output TEMP image


// preapre devices
char device[100] = "";
init(ccdsd_data, 0,0, device);
// computation process
device_memory_init(ccdsd_data);
// encode part
if(generate)
{
    unsigned int h_size_padded = ccdsd_data->h_size + ccdsd_data->pad_rows;
    unsigned int w_size_padded = ccdsd_data->w_size + ccdsd_data->pad_columns;
    int  *data_bw = NULL;
    data_bw = (int*)calloc((h_size_padded ) * (w_size_padded), sizeof(int));
    for (unsigned int i = 0; i < h_size_padded * w_size_padded; ++ i)
	{
		data_bw[i] = rand() % (NUMBER_BASE);
	}
    encode_engine(ccdsd_data, data_bw);
}
else
{
encode_engine(ccdsd_data, readBMP(ccdsd_data->filename_input, (ccdsd_data->h_size + ccdsd_data->pad_rows), (ccdsd_data->w_size + ccdsd_data->pad_columns)));
}

// copy memory
get_elapsed_time(ccdsd_data,false);
// clean memory
clean(ccdsd_data);
    
}

int* readBMP(char* filename, unsigned int pad_rows, unsigned int pad_columns)
{
	BMP Image;
   	Image.ReadFromFile( filename );
    int size_output =  Image.TellWidth() * Image.TellHeight();
	int  *data_bw = NULL;
	data_bw = (int*)calloc((Image.TellHeight() + pad_rows ) * (Image.TellWidth() + pad_columns), sizeof(int));
   	// convert each pixel to greyscale
   	for( int i=0 ; i < Image.TellHeight() ; i++)
   	{
    	for( int j=0 ; j < Image.TellWidth() ; j++)
    	{
			data_bw[i * Image.TellWidth() + j] =  (Image(j,i)->Red + Image(j,i)->Green + Image(j,i)->Blue)/3;
    	}
   }
   //we need to duplicate rows and columns to be in BLOCKSIZEIMAGE
   for(unsigned int i = 0; i < pad_rows ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_columns; j++)
			data_bw[(i + Image.TellHeight()) * Image.TellWidth() + j] = data_bw[(Image.TellHeight() - 1)* Image.TellWidth() + j];
	}

	for(unsigned int i = 0; i < pad_columns ; i++)
	{
		for(unsigned int j = 0; j < Image.TellWidth() + pad_rows ; j++)
			data_bw[(j)* Image.TellWidth() + (i + Image.TellWidth())] = data_bw[j * Image.TellWidth() + ( Image.TellWidth() - 1)];
	}
    
   return data_bw;
}

void print_usage(const char * appName)
{

    printf("Usage: %s [-e] [filename to encode decode] -o [filename for the output] -w [size] -h [size]  [-t [type]]\n", appName);
	printf(" -e filename : file to be encoded\n");
    printf(" -o filename : filename for the output data\n");
    printf(" -g : generated image\n");
    printf(" -w size : width of the input image in pixels\n");
	printf(" -h size : height of the input image in pixels \n");
	printf(" -t type: tipe of conversion i for interger (default) and f for float\n");
}


int arguments_handler(int argc, char ** argv,unsigned int *w_size, unsigned int *h_size, bool *encode, bool *generate ,bool *type, char *input_image, char *output_image){
    if (argc < 6){
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    bool selected = false;
    char type_data[1];
	for(unsigned int args = 1; args < argc; ++args)
	{
        switch (argv[args][1]) {
            case 'e' : args +=1;
                       if (!selected){ 
                            *encode = true;
                            selected = true;
					        strcpy(input_image ,argv[args]);
                        }
                        else{
                            printf("Only is possible to select -e or -g not both at the same time\n\n");
                            print_usage(argv[0]);
                            return ERROR_ARGUMENTS;
                        }
					   break;
            case 'g' : 
                        if (!selected){ 
                            *generate= true;
                            selected = true;
                        }
                        else{
                            printf("Only is possible to select -e or -g not both at the same time\n\n");
                            print_usage(argv[0]);
                            return ERROR_ARGUMENTS;
                        }
					   break;
            case 'w' : args +=1; *w_size = atoi(argv[args]);break;
            case 'h' : args +=1; *h_size = atoi(argv[args]);break;
            case 't' : args +=1; strcpy(type_data,argv[args]);break;
            case 'o' : args +=1; strcpy(output_image ,argv[args]);break;
			default: print_usage(argv[0]); return ERROR_ARGUMENTS;
		}

    }
    if ( *w_size <= IMAGE_WIDTH_MIN){
		printf("-w need to be set and bigger than or equal to 32\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    if ( *h_size <= IMAGE_HEIGHT_MIN){
		printf("-h need to be set and bigger than or equal to 32\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
	}
    if(!strcmp(type_data,"i")){ // TODO FIX
        *type = false;
    }
    else if(!strcmp(type_data, "f")){
        *type = true;
    }

    if (!selected) {
        printf("-e // -g need to be set\n\n");
		print_usage(argv[0]);
		return ERROR_ARGUMENTS;
    }
	return OK_ARGUMENTS;
}

/*void  writeBMP(char* filename, int* output ,unsigned int w_size, unsigned int h_size){
    float  *aux_data = NULL;
	aux_data = (float*)calloc(h_size * w_size, sizeof(float));
    for(unsigned int i = 0; i < h_size * w_size;++i)
    {
         aux_data[i] = float(output[i]);
    }
    writeBMP(filename, aux_data ,w_size, h_size);
    free(aux_data);
}*/
void writeBMP(char* filename, int* output ,unsigned int w_size, unsigned int h_size){

    BMP output_image;
    output_image.SetSize(w_size, h_size);
    output_image.SetBitDepth(32);
    for(unsigned int i=0; i<h_size; ++i){
        for(unsigned int j=0; j<w_size; ++j){
            int a = output [i * w_size +j];
            printf("%d ",a);
            output_image(j,i)->Blue=int(a); 
            output_image(j,i)->Red=int(a);
            output_image(j,i)->Green=int(a );
        }
        printf("\n");
    }
    output_image.WriteToFile(filename);
}