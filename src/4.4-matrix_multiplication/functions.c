// Matrix multiplication: C = A * B
// A [n*m] B [m*w] C [n w]
// C elements are expected to be initially equal to zero
void matrix_multiplication(const double* A, const double* B, double* C, const unsigned int n, const unsigned int m, const unsigned int w ){
	for (unsigned int i = 0; i < n; ++i){
		for (unsigned int j = 0; j < w; ++j){
			for (unsigned int k = 0; k < m; ++k){
				C[i*n+j] = C[i*n+j] + A[i*n+k] * B[k*w+j];
			}
		}
	}
}


//==============================================================
double* pA;
double* pB;
void readInputMatrixes(char* _inFile){
	FILE *f;
	double D;
	int N, i;

	f=fopen(_inFile, "r+b");
	if(f == NULL){
		printf("Error opening file: %s\n", inFile);
		exit(1);
	}
	readDouble(&D, f);
	N = (int)D;
	printf("N = %d\n", N);

	pA    = (double*)malloc(N*N*sizeof(double));
	pB    = (double*)malloc(N*N*sizeof(double));
	for(i=0; i<N*N; i++){
		readDouble(&D, f);
		pA[i] = D;
	}
	for(i=0; i<N*N; i++){
		readDouble(&D, f);
		pB[i] = D;
	}
	fclose(f);
}



//==============================================================
void readDouble(double *_d, FILE* _f){
	double locD;
	int res;
	res = fread(&locD, sizeof(double), 1, _f);
	if(res != 1){
		printf("readDouble error\n");
		exit(1);
	}
#ifdef __BIG_ENDIAN__
	unsigned char *r, *w;
	int i;
	r = (unsigned char*)&locD;
	w = (unsigned char*)_d;
	for(i=0; i<8; i++){
		w[i] = r[7-i];
	}
#else
	*_d = locD;
#endif
}


//==============================================================
void writeDouble(double *_d, FILE* _f){
	double locD;
	int res;

#ifdef __BIG_ENDIAN__
	unsigned char *r, *w;
	int i;
	r = (unsigned char*)_d;
	w = (unsigned char*)&locD;
	for(i=0; i<8; i++){
		w[i] = r[7-i];
	}
#else
	locD = *_d;
#endif
	res =fwrite((void*)&locD, sizeof(double), 1, _f);
	if(res != 1){
		printf("writeDouble error (%d)\n", res);
		exit(1);
	}
}

//==============================================================

