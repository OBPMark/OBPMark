#htvar kernel_code
void kernel binary_reverse_kernel(global const bench_t* B, global bench_t* Br, const long int size, const unsigned int group){     
       int id = get_global_id(0);                                                                                                  
       unsigned int position = 0;                                                                                                  
                                                                                                                                   
       if (id < size)                                                                                                              
       {                                                                                                                           
                                                                                                                                   
           unsigned int j = id;                                                                                                    
           j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
           j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
           j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
           j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
           j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
           j >>= (32-group);                                                                                                       
           position = j * 2;                                                                                                       
                                                                                                                                   
           Br[position] = B[id *2];                                                                                                
           Br[position + 1] = B[id *2 + 1];                                                                                        
       }                                                                                                                           
                                                                                                                                   
                                                                                                                                   
                                                                                                                                   
}                                                                                                                                  
void kernel fft_kernel(global bench_t* B, const int loop, const bench_t wpr ,const bench_t wpi, const unsigned int theads){        
                                                                                                                                   
           bench_t tempr, tempi;                                                                                                   
           unsigned int i = get_global_id(0);                                                                                      
           unsigned int j;                                                                                                         
           unsigned int inner_loop;                                                                                                
           unsigned int subset;                                                                                                    
           unsigned int id;                                                                                                           
                                                                                                                                   
           bench_t wr = 1.0;                                                                                                       
           bench_t wi = 0.0;                                                                                                       
           bench_t wtemp = 0.0;                                                                                                    
                                                                                                                                   
           subset = theads / loop;                                                                                                 
           id = i % subset;                                                                                                        
           inner_loop = i / subset;                                                                                                
           //get wr and wi                                                                                                         
           for(unsigned int z = 0; z < inner_loop ; ++z){                                                                          
                  wtemp=wr;                                                                                                        
                  wr += wr*wpr - wi*wpi;                                                                                           
                  wi += wi*wpr + wtemp*wpi;                                                                                        
           }                                                                                                                       
           // get I                                                                                                                
           i = id *(loop * 2 * 2) + 1 + (inner_loop * 2);                                                                          
           j=i+(loop * 2 );                                                                                                        
           tempr = wr*B[j-1] - wi*B[j];                                                                                            
           tempi = wr * B[j] + wi*B[j-1];                                                                                          
           B[j-1] = B[i-1] - tempr;                                                                                                   
           B[j] = B[i] - tempi;                                                                                                    
           B[i-1] += tempr;                                                                                                           
           B[i] += tempi;                                                                                                          
}
#htendvar