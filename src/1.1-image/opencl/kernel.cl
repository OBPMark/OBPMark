#htvar kernel_code
void kernel image_offset_correlation_gain_correction(global const int *image_input,global const int *correlation_table,global const int *gain_correlation_map, global int *processing_image, const int size_image)
{   
    unsigned int x = get_global_id(0);	
    if (x < size_image)
    {
        processing_image[x] =(image_input[x] - correlation_table[x]) * gain_correlation_map[x];
    }
}
void kernel bad_pixel_correlation(global int *processing_image,global int *processing_image_error_free, global const bool *bad_pixel_map, const unsigned int w_size ,const unsigned int h_size)
{
    unsigned int x =  get_global_id(0);
    unsigned int y =  get_global_id(1);

        if (x < w_size && y < h_size )
    {
        if (bad_pixel_map[y * h_size + x])
        {
            if (x == 0 && y == 0)
            {
                // TOP left
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x +1)] + processing_image[(y +1) * h_size +  (x +1) ] + processing_image[(y +1) * h_size + x  ])/3;
            }
            else if (x == 0 && y == h_size)
            {
                // Top right
                processing_image_error_free[y * h_size + x] = (processing_image[y * h_size +  (x -1)] + processing_image[(y -1) * h_size +  (x -1)] + processing_image[(y -1) * h_size + x ])/3;
            }
            else if(x == w_size && y == 0)
            {
                //Bottom left
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x] + processing_image[(y -1) * h_size +  (x + 1)] + processing_image[y * h_size +  (x +1)])/3;
            }
            else if (x == w_size && y == h_size)
            {
                // Bottom right
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  (x -1)] + processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1)])/3;
            }
            else if (y == 0)
            {
                // Top Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x -1) ] + processing_image[y * h_size +  (x +1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (x == 0)
            {
                //  Left Edge
                processing_image_error_free[y * h_size + x] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x +1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (x == w_size)
            {
                //  Right Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1) ] + processing_image[(y +1) * h_size +  x ])/3;
            }
            else if (y == h_size)
            {
                // Bottom Edge
                processing_image_error_free[y * h_size + x ] = (processing_image[(y -1) * h_size +  x ] + processing_image[y * h_size +  (x -1) ] + processing_image[y * h_size +  (x +1)])/3;
            }
            else
            {
                // Standart Case
                processing_image_error_free[y * h_size + x ] = (processing_image[y * h_size +  (x -1)] + processing_image[y * h_size +  (x -1) ] + processing_image[(y +1) * h_size +  x  ] +  processing_image[(y +1) * h_size +  x  ])/4;
            }
        }
        else{
            processing_image_error_free[y * h_size + x ] = processing_image[y * h_size + x];
        }

    }
}
void kernel spatial_binning_temporal_binning(global const int *processing_image,global int *output_image, const unsigned int w_size_half ,const unsigned int h_size_half)
{
    unsigned int x =  get_global_id(0);
    unsigned int y =  get_global_id(1);
    if (x < w_size_half && y < h_size_half )
    {
        output_image[y * h_size_half + x ] += processing_image[ (2*y)* (h_size_half*2) + (2 *x) ] + processing_image[(2*y)* (h_size_half*2) + (2 *(x+1))  ] + processing_image[(2*(y+1))* (h_size_half*2) + (2 *x) ] + processing_image[(2*(y+1))* (h_size_half*2) + (2 *(x+1)) ];
    }
}
#htendvar
