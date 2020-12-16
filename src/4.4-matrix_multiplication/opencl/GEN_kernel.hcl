
std::string kernel_code = 
"void kernel kernel_matrix_multiplication(global const bench_t* A, const global bench_t* B, global bench_t* C, const int n, const int m, const int w ){\n"
"int i = get_global_id(0);\n"
"int j = get_global_id(1);\n"
"if (i < n && j < w){\n"
"bench_t acumulated = 0;\n"
"for (unsigned int k_d = 0; k_d < m; ++k_d )\n"
"{\n"
"acumulated += A[i*n+k_d] * B[k_d*w +j];\n"
"}\n"
"C[i*n+j] =  acumulated;\n"
"}\n"
"}\n"
;
