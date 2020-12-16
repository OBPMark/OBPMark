#htvar kernel_code
void kernel kernel_matrix_multiplication(global const bench_t* A, const global bench_t* B, global bench_t* C, const int n, const int m, const int w,const int BLOCK_SIZE){	
                                                                                                                                                                       
       __local bench_t A_tile[16*16];                                                                                                                                                             
       __local bench_t B_tile[16*16];                                                                                                                                                               
		unsigned int i = get_group_id(0) * BLOCK_SIZE + get_local_id(0);																								
 		unsigned int j = get_group_id(1) * BLOCK_SIZE + get_local_id(1);                                                                                                
       unsigned int theadx = get_local_id(0);                                                                                                                          							
       unsigned int theady = get_local_id(1);                                                                                                                          
                                                                                                                                                                       
       bench_t acumulated = 0;                                                                                                                                           
       unsigned int idx = 0;                                                                                                                                           
																	 													                
       for (unsigned int sub = 0; sub < get_num_groups(0); ++sub)                                                                                                      
       {																													                                            
			idx = i * n + sub * BLOCK_SIZE + theady;										 														                    
			if(idx >= m*n)											             																				        
			{    											             										                                                        
                A_tile[theadx * BLOCK_SIZE+ theady] = 0;   									                             																	    
            }																													                                        
            else                                                                                                                                                        
           {                                                                                                                                                           
               A_tile[theadx * BLOCK_SIZE + theady] = A[idx];                                                                                                          
           }                                                                                                                                                           
           idx = (sub * BLOCK_SIZE + theadx) * w + j;                                                                                                                  
           if (idx >= m*w)                                                                                                                                             
           {                                                                                                                                                           
               B_tile[theadx * BLOCK_SIZE +  theady] = 0;                                                                                                              
           }                                                                                                                                                           
           else                                                                                                                                                        
           {                                                                                                                                                           
               B_tile[theadx* BLOCK_SIZE + theady] = B[idx];                                                                                                           
           }                                                                                                                                                           
           barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                               
           for (unsigned int k = 0; k < BLOCK_SIZE; ++k)                                                                                                               
           {                                                                                                                                                           
               acumulated +=  A_tile[theadx*BLOCK_SIZE + k] * B_tile[k*BLOCK_SIZE + theady];                                                                           
           }                                                                                                                                                           
           barrier(CLK_LOCAL_MEM_FENCE);                                                                                                                               
       }                                                                                                                                                               
       if (i < n && j < w)                                                                                                                                             
       {                                                                                                                                                               
           C[i *n + j] = acumulated;                                                                                                                                   
       }                                                                                                                                                               
}
#htendvar																				 																	                    