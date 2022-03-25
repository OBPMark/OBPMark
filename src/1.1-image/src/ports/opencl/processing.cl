#htvar kernel_code

// NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
// so uint16_t will be switch to unsigned short
// and uint32_t  will be switch to unsigned int

void kernel
f_offset(
    global unsigned short *frame,
    const int i,
    global const unsigned short *offsets,
    const int size
)
{

    unsigned short *frame_i;
    frame_i = frame + (size * i);
    unsigned int x = get_global_id(0);	
    if (x < size)
    {
        frame_i[x] = frame_i[x] - offsets[x];
    }
}

void kernel
f_mask_replace(
    global unsigned short *frame,
    const int i,
    global const unsigned char *mask,
    const unsigned int width,
    const unsigned int height
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int kernel_rad = 1;
    unsigned int n_sum = 0;
    unsigned int sum = 0;

    unsigned short *frame_i;
    frame_i = frame + (width * height  * i);

     if (x < width && y < height)
    {
        if (mask[y* width + x] == 1)
        {
            for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3 
                {
                    for(int j = -kernel_rad; j <= kernel_rad; ++j){
                        
                        if (!((i + x < 0 || j + y < 0) || ( i + x > height - 1 || j + y > width - 1)))
                        {
                            //printf("POS s x %d y %d value %d\n", y + j, x + i, (y + j)* width + (x + i));
                            if ( mask[(y + j)* width + (x + i)] == 0)
                            {
                                sum += frame_i[(y + j)*width+(x + i)]; 
                                ++n_sum;
                            }
                            
                        }
                        
                    }
                }
                
            frame_i[y * width + x] = (unsigned short)(n_sum == 0 ? 0 : sum / n_sum);

        }
        //printf("POS s x %d y %d value %d\n", threadIdx.x, threadIdx.y, frame[y * width + x]);
    }
}

void kernel
f_scrub(
    global unsigned short *frame,
    const unsigned int frame_i,
    const unsigned int frame_i_0,
    const unsigned int frame_i_1,
    const unsigned int frame_i_2,
    const unsigned int frame_i_3,
    const unsigned int width,
    const unsigned int height
    )
{
    const unsigned int num_neighbour = 4;
	
	unsigned int sum;
	unsigned int mean;
	unsigned int thr;

    unsigned short *frame_0;
    unsigned short *frame_1;
    unsigned short *frame_2;
    unsigned short *frame_3;
    unsigned short *frame_i_point;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height)
    {
        /* Init Frames position */
        frame_0 = frame + ((height * width) * frame_i_0);
        frame_1 = frame + ((height * width) * frame_i_1);
        frame_2 = frame + ((height * width) * frame_i_2);
        frame_3 = frame + ((height * width) * frame_i_3);
        frame_i_point = frame + ((height * width) * frame_i);

        
        sum = (unsigned short)frame_0[y * width + x] + (unsigned short)frame_1[y * width + x] + (unsigned short)frame_2[y * width + x] + (unsigned short)frame_3[y * width + x];
        /* Calculate mean and threshold */
        mean = sum / (num_neighbour); 
        thr = 2*mean; 
        /* If above threshold, replace with mean of temporal neighbours */
        if (frame_i_point[y * width + x] > thr)
        {
            // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
            // so uint16_t will be switch to unsigned short
            // and uint32_t  will be switch to unsigned int
            frame_i_point[y * width + x] = (unsigned short)mean;
        }
    }

}

void kernel
f_gain(
    global unsigned short *frame,
    const unsigned int i,
    global unsigned short *gains,
    const unsigned int size,
    const unsigned int width,
    const unsigned int height
    )
{
    unsigned short *frame_i;
    int x = get_global_id(0);
    frame_i = frame + ((height * width) * i);

    if (x < size)
    {
        // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
        // so uint16_t will be switch to unsigned short
        // and uint32_t  will be switch to unsigned int
        frame_i[x] = (unsigned short)((unsigned int)frame_i[x] * (unsigned int)gains[x] >> 16 );
    }
}

void kernel
f_2x2_bin_coadd(
    global unsigned short *frame,
    const unsigned int i_frame,
    global unsigned int *sum_frame,
    const unsigned int width,
    const unsigned int height,
    const unsigned int lateral_stride
    )
{
    const unsigned int stride = 2;
    unsigned int sum = 0;
    unsigned short *frame_i;
    int i = get_global_id(0);
    int j = get_global_id(1);
    frame_i = frame + (height * 2 * width * 2 * i_frame);

    if (i < width && j < height){
        #pragma unroll
        for(unsigned int x = 0; x < stride; ++x)
        {
            #pragma unroll
            for(unsigned int y = 0; y < stride; ++y)
            {   
                // NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
                // so uint16_t will be switch to unsigned short
                // and uint32_t  will be switch to unsigned int
                sum +=  (unsigned short)frame_i[((j * stride) + x) * (width * stride) + ((i*stride) +y)];
                
            }
        }
        sum_frame[j * lateral_stride + i ]= sum + sum_frame[j * lateral_stride + i ];
    }
}

#htendvar
