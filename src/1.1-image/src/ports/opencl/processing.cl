#htvar kernel_code

void kernel
f_offset(
    global uint16_t *frame,
    global const uint16_t *offsets,
    const int size
)
{
    unsigned int x = get_global_id(0);	
    if (x < size)
    {
        frame[x] = frame[x] - offsets[x];
    }
}

void kernel
f_mask_replace(
    global uint16_t *frame,
    global const uint16_t *mask,
    const unsigned int width,
    const unsigned int height
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
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

void kernel
f_scrub(
    global uint16_t *frame,
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
	
	uint32_t sum;
	uint32_t mean;
	uint32_t thr;

    uint16_t *frame_0;
    uint16_t *frame_1;
    uint16_t *frame_2;
    uint16_t *frame_3;
    uint16_t *frame_i_point;

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

        
        sum = frame_0[y * width + x] + frame_1[y * width + x] + frame_2[y * width + x] + frame_3[y * width + x];
        /* Calculate mean and threshold */
        mean = sum / (num_neighbour); 
        //thr = 2*mean; 
        /* If above threshold, replace with mean of temporal neighbours */
        if (frame[y * width + x] > thr)
        {
            frame_i_point[y * width + x] = (uint16_t)mean;
        }
    }

}

void kernel
f_gain(
    global uint16_t *frame,
    global uint16_t *gains,
    const unsigned int size
    )
{
    int x = get_global_id(0);

    if (x < size)
    {
        frame[x] = (uint16_t)((uint32_t)frame[x] * (uint32_t)gains[x] >> 16 );
    }
}

void kernel
f_2x2_bin_coadd(
    global uint16_t *frame,
    global uint32_t *sum_frame,
    const unsigned int width,
    const unsigned int height,
    const unsigned int lateral_stride
    )
{
    const unsigned int stride = 2;
    uint32_t sum = 0;
    int i = get_global_id(0);
    int j = get_global_id(1);

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

#htendvar
