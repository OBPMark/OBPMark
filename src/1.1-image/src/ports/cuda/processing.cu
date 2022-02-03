/**
 * \file processing.c
 * \brief Benchmark #1.1 CUDA kernel implementation.
 * \author Ivan Rodriguez-Ferrandez (BSC)
 */
 #include "device.h"
 #include "processing.h"
 #include "obpmark.h"
 #include "obpmark_time.h"
 
///////////////////////////////////////////////////////////////////////////////////////////////
// KERNELS
///////////////////////////////////////////////////////////////////////////////////////////////


/*__global__
void f_mask_replace_offset(
	uint16_t *frame,
	uint8_t *mask,
    const uint16_t *offsets,
    const unsigned int width,
    const unsigned int height,
    const int shared_size
	)
{
    const int kernel_rad = 1;
    unsigned int n_sum = 0;
    uint32_t sum = 0;
    int x0, y0;
    extern __shared__ int data[];

	int x = blockIdx.x * blockDim.x + threadIdx.x; // Should be unsigned int, but if I put unsigned I get a warning for checking against 0 in the if
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
    if (x < width && y < height)
    {
        // each thread load 4 values ,the corners
        //TOP right corner
        x0 = x - kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0 || y0 < 0 )
        {
            data[threadIdx.x * shared_size + threadIdx.y] = -1;
        }
        else
        {
            //printf("POS x %d y %d x0 %d y0 %d, %d\n", threadIdx.x, threadIdx.y, x0, y0, frame[0]);
            data[threadIdx.x * shared_size + threadIdx.y] = frame[y0 *width+x0] - offsets[y0 * width + x0];
        } 
            
        //BOTTOM right corner
        x0 = x + kernel_rad;
        y0 = y - kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > height-1  || y0 < 0 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = -1;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + threadIdx.y] = frame[y0 *width+x0] - offsets[y0 * width + x0];
        } 

        //TOP left corner
        x0 = x - kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 < 0  || y0 > width-1 )
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = -1;
        }
        else
        {
            data[threadIdx.x * shared_size + (threadIdx.y + kernel_rad * 2)] = frame[y0 *width+x0] - offsets[y0 * width + x0];
        } 

        //BOTTOM left corner
        x0 = x + kernel_rad;
        y0 = y + kernel_rad;
        //printf("POS x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, x0, y0);
        if ( x0 > height-1  || y0 > width-1 )
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = -1;
        }
        else
        {
            data[(threadIdx.x + kernel_rad * 2) * shared_size + (threadIdx.y + kernel_rad * 2)] = frame[y0 *width+x0] - offsets[y0 * width + x0];
        } 
        // finish loading data
        __syncthreads();

        unsigned int xa = kernel_rad + threadIdx.x;
        unsigned int ya = kernel_rad + threadIdx.y;

        if (mask[y* width + x] == 0)
        {
            // the pixel is valid store the valid data
            frame[y* width + x] = (uint16_t)data[xa * shared_size +  ya];

        }
        else
        {   
            // the pixel is not valid
            #pragma unroll
            for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
                {
                    #pragma unroll
                    for(int j = -kernel_rad; j <= kernel_rad; ++j)
                    {   

                        printf("POS s x %d y %d x0 %d y0 %d\n", threadIdx.x, threadIdx.y, (xa + i), (ya + j));
                        if ( data[(xa + i) * shared_size +  (ya + j)] != -1 ) // check if the value is valid 
                        {
                            printf("POS x %d y %d x0 %d y0 %d\n", x, y, (x + j), (y + i));
                            //if (mask[(y + i)* width + (x + j)] == 0) //  check if the value is not a bad pixel
                            //{
                                //sum += data[(xa + i) * shared_size +  (ya + j)];
                                //n_sum++;
                            //}
                            
                        }
                        
                    }
                }
                    
            frame[y * width + x] = (uint16_t)(n_sum == 0 ? 0 : sum / n_sum);
        }
        
    }
		
}*/

__global__
void f_offset(
	uint16_t *frame,
	const uint16_t *offsets,
    const int size	
	)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size)
    {
        frame[x] = frame[x] - offsets[x];
        //printf("POS  x %d  value %d\n", x, frame[x]);
    }

}

__global__
void f_mask_replace(
	uint16_t *frame,
	const uint8_t *mask,
    const unsigned int width,
    const unsigned int height
	)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x; // Should be unsigned int, but if I put unsigned I get a warning for checking against 0 in the if
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int kernel_rad = 1;
    unsigned int n_sum = 0;
    uint32_t sum = 0;

    if (x < width && y < height)
    {
        if (mask[y* width + x] != 0)
        {
            for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
                {
                    for(int j = -kernel_rad; j <= kernel_rad; ++j){
                        if (!(i + x < 0 || j + y < 0) || !( i + x > height - 1 || j + y > width - 1))
                        {
                            if ( mask[(y + j)* width + (x + i)] == 0)
                            {
                                sum += frame[(x + i)*width+(y + j)];
                                ++n_sum;
                            }
                            
                        }
                        
                    }
                }
                
            frame[y * width + x] = (uint16_t)(n_sum == 0 ? 0 : sum / n_sum);

        }
        //printf("POS s x %d y %d value %d\n", threadIdx.x, threadIdx.y, frame[y * width + x]);
    }
}


__global__
void f_scrub(
	uint16_t *frame,
	uint16_t *frame_i_0,
    uint16_t *frame_i_1,
    uint16_t *frame_i_2,
    uint16_t *frame_i_3,
    const unsigned int width,
    const unsigned int height
	)
{
	static unsigned int num_neighbour = 4;
	
	uint32_t sum;
	uint32_t mean;
	uint32_t thr;

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        /* Generate scrubbing mask */
        /* Sum temporal neighbours between -2 to + 2 without actual */ // FIXME unroll by 4
        sum = frame_i_0[y * width + x] + frame_i_1[y * width + x] + frame_i_2[y * width + x] + frame_i_3[y * width + x];
        //sum = fs[(y * width + x) + ((width * height) * frame_i_0)] + fs[(y * width + x) + ((width * height) * frame_i_1)] + fs[(y * width + x) + ((width * height) * frame_i_2)] + fs[(y * width + x) + ((width * height) * frame_i_3)];
        //sum = fs[frame_i_0].f[y * width + x] + fs[frame_i_1].f[y * width + x] + fs[frame_1_2].f[y * width + x] + fs[frame_i_3].f[y * width + x];
        /* Calculate mean and threshold */
        mean = sum / (num_neighbour); 
        thr = 2*mean; 
        /* If above threshold, replace with mean of temporal neighbours */
        if (frame[y * width + x] > thr)
        {
            frame[y * width + x] = (uint16_t)mean;
        }
    }		
	
}
__global__
void f_gain(
	uint16_t *frame,
	uint16_t *gains,
    const int size	
	)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < size)
    {
        frame[x] = (uint16_t)((uint32_t)frame[x] * (uint32_t)gains[x] >> 16 );
    }

}

__global__
void f_2x2_bin_coadd(
	uint16_t *frame,
	uint32_t *sum_frame,
    const unsigned int width,
    const unsigned int height,
    const unsigned int lateral_stride
	)
{
    const unsigned int stride = 2;
    uint32_t sum = 0;
    

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height){
        #pragma unroll
        for(unsigned int x = 0; x < stride; ++x)
        {
            #pragma unroll
            for(unsigned int y = 0; y < stride; ++y)
            {
                sum +=  frame[((j * stride) + x) * width + ((i*stride) +y)];
                
            }
        }
        sum_frame[j * lateral_stride + i ]= sum + sum_frame[j * lateral_stride + i ];
    }
}


