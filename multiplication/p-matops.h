#define ALG_SUM(op, a, b) ((op) ? ((a)-(b)) : ((a)+(b)))
#define IS_POWTWO(x) ( ( ((x) != (0)) && ( ((x) & (x - 1)) == (0) ) ) ? (1) : (0) )

#ifndef MAX
#define MAX(a, b) ( ((a) > (b)) ? (a) : (b) )
#endif

struct fmat_t *msum(struct fmat_t *A, struct fmat_t *B, int issub, int threads);
struct fmat_t *mprod_rowcol(struct fmat_t *A, struct fmat_t *B, int threads);
struct fmat_t *mprod_strassen(struct fmat_t *A, struct fmat_t *B, int threads);
struct fmat_t *mprod_task(struct fmat_t *A, struct fmat_t *B, int block_size, int threads);

/****************************************************************************
 * Source code
 * *************************************************************************/

struct 
fmat_t *msum(struct fmat_t *A, struct fmat_t *B, int issub, int threads) {
	
	if ((A == NULL) || (B == NULL)) {return NULL;}
        
        if (!(A->size1 == B->size1) && 
			(A->size2 == B->size2)) {
                printf("ERROR: Dimensions mismatch.\n");
                printf("Cannot perform matrix summation.\n");
                return NULL;
        }

	
	struct fmat_t *C;
	C = f_matrix_alloc(A->size1, A->size2);
	if (C == NULL) {return NULL;}

	float *dda, *ddb, *ddc;
	dda = A->data;
	ddb = B->data;
	ddc = C->data;
	
	int chunks = MAX( 1, (int)(ceil(A->block->size / threads)) );
	int i, j;
	#pragma omp parallel for \
	private(i, j) shared(A, B, C, dda, ddb, ddc, chunks) \
	collapse(2) schedule(dynamic, chunks)
	for (i = 0; i < A->size1; i++) {
		for (j = 0; j < A->size2; j++){
			ddc[i * C->tda + j] = ALG_SUM(issub, 
						      dda[i * A->tda + j], 
						      ddb[i * B->tda + j]);
		}
	}
	
	return C;
}


struct 
fmat_t *mprod_rowcol(struct fmat_t *A,
		     struct fmat_t *B,
		     int threads) {
	
	if ((A == NULL) || (B == NULL)) {return NULL;}
	
	if (A->size2 != B->size1) {
		printf("Dimensions mismatch.\n");
		printf("Cannot perform matrix multiplication.\n");
		return NULL;
	}
	struct fmat_t *C;
	C = f_matrix_calloc(A->size1, B->size2, threads);
	if (C == NULL) {
		return NULL;
	}
	
	float *dda, *ddb, *ddc;
	dda = A->data;
	ddb = B->data;
	ddc = C->data;
	
	int chunks = MAX( 1, (int)(ceil(B->size2 / threads)) );
	int i, j, k;
	for (i = 0; i < A->size1; i++) {
		#pragma omp parallel for \
		private(j, k) shared(A, B, C, dda, ddb, ddc, chunks)\
		schedule(dynamic, chunks)
		for (j = 0; j < B->size2; j++) {
			for (k = 0; k < B->size1; k++) {
				ddc[i * C->tda + j] += dda[i * A->tda + k] * ddb[k * B->tda + j];
			}
		}
	}
	return C;
}


struct
fmat_t *mprod_strassen(struct fmat_t *A,
		       struct fmat_t *B,
		       int threads) {
	if (!((A->size1 == A->size2) && 
	   (A->size1 == B->size1) &&
	   (A->size2 == B->size2))) {
		printf("Dimensions mismatch.\n");
		printf("Cannot perform matrix multiplication.\n");
		return NULL;
	}

	if (!(IS_POWTWO(A->size1))) {
		printf("Matrix must be 2^n x 2^n.\n");
		printf("Cannot perform matrix multiplication.\n");
		return NULL;
	}
	
	
	if (A->size1 <= 256) {
		return mprod_rowcol(A, B, threads);
	}
	

	struct fmat_t *A11, *A12, *A21, *A22,
		      *B11, *B12, *B21, *B22,
		      *C11, *C12, *C21, *C22,
		      *C, *tmpA, *tmpB, 
		      *P1, *P2, *P3, *P4,
		      *P5, *P6, *P7;

	size_t n = A->size1 / 2;

	#pragma omp sections
	{
		#pragma omp section
		A11 = f_matrix_submatrix(A, 0, 0, n, n);
		#pragma omp section
		A12 = f_matrix_submatrix(A, 0, n, n, n);
		#pragma omp section
		A21 = f_matrix_submatrix(A, n, 0, n, n);
		#pragma omp section
		A22 = f_matrix_submatrix(A, n, n, n, n);
		#pragma omp section
		B11 = f_matrix_submatrix(B, 0, 0, n, n);
		#pragma omp section
		B12 = f_matrix_submatrix(B, 0, n, n, n);
		#pragma omp section
		B21 = f_matrix_submatrix(B, n, 0, n, n);
		#pragma omp section
		B22 = f_matrix_submatrix(B, n, n, n, n);
	}

	/* P1 */
	tmpA = msum(A11, A22, 0, threads);
	tmpB = msum(B11, B22, 0, threads);
	P1 = mprod_strassen(tmpA, tmpB, threads);
	
	/* P2 */
	tmpA = msum(A21, A22, 0, threads);
	P2 = mprod_strassen(tmpA, B11, threads);
	
	/* P3 */
	tmpB = msum(B12, B22, 1, threads);
	P3 = mprod_strassen(A11, tmpB, threads);
	
	/* P4 */
	tmpB = msum(B21, B11, 1, threads);
	P4 = mprod_strassen(A22, tmpB, threads);
	
	/* P5 */
	tmpA = msum(A11, A12, 0, threads);
	P5 = mprod_strassen(tmpA, B22, threads);
	
	/* P6 */
	tmpA = msum(A21, A11, 1, threads);
	tmpB = msum(B11, B12, 0, threads);
	P6 = mprod_strassen(tmpA, tmpB, threads);
	
	/* P7 */
	tmpA = msum(A12, A22, 1, threads);
	tmpB = msum(B21, B22, 0, threads);
	P7 = mprod_strassen(tmpA, tmpB, threads);
	
	/* Check */
	/*f_matrix_pprint(A11);
	f_matrix_pprint(A12);
	f_matrix_pprint(A21);
	f_matrix_pprint(A22);
	f_matrix_pprint(B11);
	f_matrix_pprint(B12);
	f_matrix_pprint(B21);
	f_matrix_pprint(B22);
	f_matrix_pprint(P1);
	f_matrix_pprint(P2);
	f_matrix_pprint(P3);
	f_matrix_pprint(P4);
	f_matrix_pprint(P5);
	f_matrix_pprint(P6);
	f_matrix_pprint(P7);*/

	C11 = msum(P1, P4, 0, threads);
	C11 = msum(C11, P5, 1, threads);
	C11 = msum(C11, P7, 0, threads);
	C12 = msum(P3, P5, 0, threads);
	C21 = msum(P2, P4, 0, threads);
	C22 = msum(P1, P3, 0, threads);
	C22 = msum(C22, P2, 1, threads);
	C22 = msum(C22, P6, 0, threads);

	/*f_matrix_pprint(C11);
	f_matrix_pprint(C12);
	f_matrix_pprint(C21);
	f_matrix_pprint(C22);*/

	C = f_matrix_join(C11, C12, C21, C22);
	
	/* Free intermediate matrices */
	f_matrix_free(C11);
	f_matrix_free(C12);
	f_matrix_free(C21);
	f_matrix_free(C22);
	f_matrix_free(P7);
	f_matrix_free(P6);
	f_matrix_free(P5);
	f_matrix_free(P4);
	f_matrix_free(P3);
	f_matrix_free(P2);
	f_matrix_free(P1);

	f_matrix_free(tmpB);
	f_matrix_free(tmpA);

	f_matrix_free(B22);
	f_matrix_free(B21);
	f_matrix_free(B12);
	f_matrix_free(B11);
	f_matrix_free(A22);
	f_matrix_free(A21);
	f_matrix_free(A12);
	f_matrix_free(A11);

	return C;
}


struct
fmat_t *mprod_task(struct fmat_t *A, struct fmat_t *B, int block_size, int threads) {

	if ((A == NULL) || (B == NULL)) {return NULL;}
        if ((A->size1 != B->size1) ||  
	    (A->size2 != B->size2) || 
	    (A->size1 != A->size2) ||
       	    ((A->size1 % block_size) != 0))	{
		printf("Matrix must be square and its dimensions must be multiple of block_size.\n");
		return NULL;
	}

	struct fmat_t *C;
	C = f_matrix_calloc(A->size1, B->size2, threads);
	if (C == NULL) {
		return NULL;
	}

	float *dda, *ddb, *ddc;
	dda = A->data;
	ddb = B->data;
	ddc = C->data;

	size_t N = A->size1;
	struct fmat_t *AA, *BB, *CC;

	#pragma omp parallel
	{
	
	#pragma omp single 
	{
	
	for(int i = 0; i < N; i += block_size) {
		//printf("%d\n", i + 1);
		for(int j = 0; j < N; j += block_size) {
			for(int k = 0; k < N; k += block_size) {
				//printf("--> %d <--\n", k);
				//printf("submat at %d, %d\n", i + 1, k + 1 );
				AA = f_matrix_submatrix(A, i, k, block_size, block_size);
				BB = f_matrix_submatrix(B, k, j, block_size, block_size);
				CC = f_matrix_submatrix(C, i, j, block_size, block_size);

				float *dda, *ddb, *ddc;
				dda = AA->data;
				ddb = BB->data;
				ddc = CC->data;

				#pragma omp task depend (in: dda, ddb) depend(inout: ddc)
				{

				for (int ii = 0; ii < block_size; ii++) {
					//printf("%d\n", ii + i + 1);
					//printf("\n");
					for (int jj = 0; jj < block_size; jj++) {
						for (int kk = 0; kk < block_size; kk++) {
							//printf("AA[%d][%d] : %f * ", ii + i + 1, kk + k + 1, dda[ii * AA->tda + kk]);
							//printf(" BB[%d][%d] : %f = ", kk + k + 1, jj + j + 1, ddb[kk * BB->tda + jj]);
							//printf(" CC[%d][%d] : %f \n", ii + i + 1, jj + j + 1, ddc[ii * CC->tda + jj]);
							ddc[ii * CC->tda + jj] += dda[ii * AA->tda + kk] * ddb[kk * BB->tda + jj];
							//printf(" CC[%d][%d] : %f\n", ii + i + 1, jj + j + 1, ddc[ii * CC->tda + jj]);
						}
					}
				}

				} /* end of task */

			}
		}
	}

	} /* end of single */
	
	} /* end of parallel */

	f_matrix_free(CC);
	f_matrix_free(BB);
	f_matrix_free(AA);
	
	return C;
}
