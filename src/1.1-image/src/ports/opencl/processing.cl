#htvar kernel_code

// NOTE: in opencl the implementation of the uint16/uint32 ar not fully working and we need to use the c like types
// so uint16_t will be switch to unsigned short
// and uint32_t  will be switch to unsigned int
// and uint8_t will be switch to unsigned char

// TODO move to a top level
#define uint32_t_cl unsigned int
#define uint16_t_cl unsigned short
#define uint8_t_cl unsigned char

void kernel
f_offset(
    global uint16_t_cl *frame,
    const int i,
    global const uint16_t_cl *offsets,
    const int size
)
{

    global uint16_t_cl *frame_i;
    frame_i = frame + (size * i);
    uint32_t_cl x = get_global_id(0);	
    if (x < size)
    {
        frame_i[x] = frame_i[x] - offsets[x];
    }
}

void kernel
f_mask_replace(
    global uint16_t_cl *frame,
    const int i,
    global const uint8_t_cl *mask,
    const uint32_t_cl width,
    const uint32_t_cl height
    )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int kernel_rad = 1;
    uint32_t_cl n_sum = 0;
    uint32_t_cl sum = 0;

    global uint16_t_cl *frame_i;
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
                                sum = sum + (uint32_t_cl)(frame_i[(y + j)*width+(x + i)]); 
                                ++n_sum;
                            }
                            
                        }
                        
                    }
                }
            //if (y == 541 && x == 140){printf("POS s x %d y %d value %u:%u\n",y, x, sum, n_sum);}
            if (n_sum == 0)
            {
                frame_i[y * width + x] = 0;
            }
            else{
                frame_i[y * width + x] = (uint16_t_cl)((unsigned int)(sum / n_sum));
            }
            //frame_i[y * width + x] = (uint16_t_cl)(n_sum == 0 ? 0 : (uint32_t_cl)(sum) / (uint32_t_cl)(n_sum));

        }
        //printf("POS s x %d y %d value %d\n", threadIdx.x, threadIdx.y, frame[y * width + x]);
    }
}

void kernel
f_scrub(
    global uint16_t_cl *frame,
    const uint32_t_cl frame_i,
    const uint8_t_cl frame_i_0,
    const uint8_t_cl frame_i_1,
    const uint8_t_cl frame_i_2,
    const uint8_t_cl frame_i_3,
    const uint32_t_cl width,
    const uint32_t_cl height
    )
{
    const uint32_t_cl num_neighbour = 4;
	
	uint32_t_cl sum;
	uint32_t_cl mean;
	uint32_t_cl thr;

    global uint16_t_cl *frame_0;
    global uint16_t_cl *frame_1;
    global uint16_t_cl *frame_2;
    global uint16_t_cl *frame_3;
    global uint16_t_cl *frame_i_point;

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

        
        sum = (uint32_t_cl)(frame_0[y * width + x] + frame_1[y * width + x] + frame_2[y * width + x] + frame_3[y * width + x]);
        /* Calculate mean and threshold */
        mean = (uint32_t_cl)( sum / num_neighbour); 
        thr = 2  * mean; 
        /* If above threshold, replace with mean of temporal neighbours */
        //if (y == 541 && x == 140){printf("%u POS s x %d y %d value %u:%u ---- %u %u %u %u\n", frame_i_3, y, x, thr, mean, frame_0[y * width + x],frame_1[y * width + x],frame_2[y * width + x],frame_3[y * width + x]);}
        if (frame_i_point[y * width + x] > thr)
        {
            //printf("POS s x %d y %d value %u:%u\n", y, x, thr, mean);
            frame_i_point[y * width + x] = (uint16_t_cl)mean;
        }
    }

}

void kernel
f_gain(
    global uint16_t_cl *frame,
    const uint32_t_cl i,
    global uint16_t_cl *gains,
    const uint32_t_cl size,
    const uint32_t_cl width,
    const uint32_t_cl height
    )
{
    global uint16_t_cl *frame_i;
    int x = get_global_id(0);
    frame_i = frame + ((height * width) * i);

    if (x < size)
    {
        frame_i[x] = (uint16_t_cl)((uint32_t_cl)frame_i[x] * (uint32_t_cl)gains[x] >> 16 );
    }
}

void kernel
f_2x2_bin_coadd(
    global uint16_t_cl *frame,
    const uint32_t_cl i_frame,
    global uint32_t_cl *sum_frame,
    const uint32_t_cl width,
    const uint32_t_cl height,
    const uint32_t_cl lateral_stride
    )
{
    const uint32_t_cl stride = 2;
    uint32_t_cl sum = 0;
    global uint16_t_cl *frame_i;
    int i = get_global_id(0);
    int j = get_global_id(1);
    frame_i = frame + (height * 2 * width * 2 * i_frame);

    if (i < width && j < height){
        #pragma unroll
        for(uint32_t_cl x = 0; x < stride; ++x)
        {
            #pragma unroll
            for(uint32_t_cl y = 0; y < stride; ++y)
            {   

                sum +=  (uint16_t_cl)frame_i[((j * stride) + x) * (width * stride) + ((i*stride) +y)];
                
            }
        }
        sum_frame[j * lateral_stride + i ]= sum + sum_frame[j * lateral_stride + i ];
    }
}

#htendvar
