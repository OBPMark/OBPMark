#htvar kernel_code
void kernel kernel_matrix_multiplication(global const bench_t* A, const global bench_t* B, global bench_t* C, const int n, const int m, const int w ){	
		int i = get_global_id(0);																							
 		int j = get_global_id(1);																				    		
		if (i < n && j < w){																	 							
       	bench_t acumulated = 0;																							
			for (unsigned int k_d = 0; k_d < m; ++k_d )										 								
				{										             														
			    	acumulated += A[i*n+k_d] * B[k_d*w +j];										             				 
			    }									                             												
				C[i*n+j] =  acumulated;																						
		}										                             													
                                           																				
}																				 											
#htendvar