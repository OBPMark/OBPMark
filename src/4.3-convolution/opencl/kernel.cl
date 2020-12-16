#htvar kernel_code
void kernel kernel_matrix_convolution(global const bench_t* A,  global bench_t* B, global const bench_t* kernel_data, const int n, const int m, const int w, const int kernel_size ){	    
		int x = get_global_id(0);																													                                    
 		int y = get_global_id(1);																				    								                                    
       unsigned int size = n;                                                                                                                                                          
       int kernel_rad = kernel_size / 2;                                                                                                                                               
       bench_t sum = 0;                                                                                                                                                                
                                                                                                                                                                                       
		if (x < size && y < size){																	 													                                
       	for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3																			
			{									 														                                                                                
				 for(int j = -kernel_rad; j <= kernel_rad; ++j)										             																	    
			     {											             										                                                                        
			    	 bench_t value = 0;								                             																	                    
                    if (i + x < 0 || j + y < 0)                                                                                                                                        
                    {                                                                                                                                                                  
                       value = 0;                                                                                                                                                      
                    }                                                                                                                                                                  
                    else if ( i + x > size - 1 || j + y > size - 1)                                                                                                                    
                    {                                                                                                                                                                  
                        value = 0;                                                                                                                                                     
                    }                                                                                                                                                                  
                    else                                                                                                                                                               
                    {                                                                                                                                                                  
                       value = A[(x + i)*size+(y + j)];                                                                                                                                
                    }                                                                                                                                                                  
                    sum += value * kernel_data[(i+kernel_rad)* kernel_size + (j+kernel_rad)];                                                                                          
                 }                                                                                                                                                                     
             }                                                                                                                                                                         
           B[x*size+y ] = sum;                                                                                                                                                         
																																                                                        
		}										                             																		                                    
                                           																										                                    
}
#htendvar