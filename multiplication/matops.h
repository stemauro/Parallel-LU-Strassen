#define ALG_SUM(op, a, b) ((op) ? ((a)-(b)) : ((a)+(b)))
#define IS_POWTWO(x) ( ( ((x) != (0)) && ( ((x) & (x - 1)) == (0) ) ) ? (1) : (0) )

struct fmat_t *msum(struct fmat_t *A, struct fmat_t *B, int issub);
struct fmat_t *mprod_rowcol(struct fmat_t *A, struct fmat_t *B);
struct fmat_t *mprod_strassen(struct fmat_t *A, struct fmat_t *B);


/****************************************************************************
 * Source code
 * *************************************************************************/

struct 
fmat_t *msum(struct fmat_t *A, struct fmat_t *B, int issub) {
	
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

	float *dda, *ddb;
	dda = A->data;
	ddb = B->data;
	
	for (int i = 0; i < A->size1; i++) {
		for (int j = 0; j < A->size2; j++){
			(C->data)[i * C->tda + j] = ALG_SUM(issub, 
							    dda[i * A->tda + j], 
							    ddb[i * B->tda + j]);
		}
	}
	
	return C;
}


struct 
fmat_t *mprod_rowcol(struct fmat_t *A,
		     struct fmat_t *B) {
	
	if ((A == NULL) || (B == NULL)) {return NULL;}
	
	if (A->size2 != B->size1) {
		printf("Dimensions mismatch.\n");
		printf("Cannot perform matrix multiplication.\n");
		return NULL;
	}
	
	struct fmat_t *C;
	C = f_matrix_calloc(A->size1, B->size2);
	if (C == NULL) {return NULL;}

	float *dda, *ddb, *ddc;
	dda = A->data;
	ddb = B->data;
	ddc = C->data;

	for (int i = 0; i < C->size1; i++) {
		for (int k = 0; k < A->size2; k++) {
			for (int j = 0; j < C->size2; j++) {
				float tmp;
				tmp = dda[i * A->tda + k] *
					ddb[k * B->tda + j];
				tmp += ddc[i * C->tda + j];
				f_matrix_insert(C, i, j, tmp);
			}
		}
	}

	return C;
}


struct
fmat_t *mprod_strassen(struct fmat_t *A,
		    struct fmat_t *B) {
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

	if (A->size1 <= 16) {
		return mprod_rowcol(A, B);
	}

	struct fmat_t *A11, *A12, *A21, *A22,
		      *B11, *B12, *B21, *B22,
		      *C11, *C12, *C21, *C22,
		      *C, *tmpA, *tmpB, 
		      *P1, *P2, *P3, *P4,
		      *P5, *P6, *P7;

	size_t n = A->size1 / 2;

	A11 = f_matrix_submatrix(A, 0, 0, n, n);
	A12 = f_matrix_submatrix(A, 0, n, n, n);
	A21 = f_matrix_submatrix(A, n, 0, n, n);
	A22 = f_matrix_submatrix(A, n, n, n, n);
	B11 = f_matrix_submatrix(B, 0, 0, n, n);
	B12 = f_matrix_submatrix(B, 0, n, n, n);
	B21 = f_matrix_submatrix(B, n, 0, n, n);
	B22 = f_matrix_submatrix(B, n, n, n, n);

	/* P1 */
	tmpA = msum(A11, A22, 0);
	tmpB = msum(B11, B22, 0);
	P1 = mprod_strassen(tmpA, tmpB);

	/* P2 */
	tmpA = msum(A21, A22, 0);
	P2 = mprod_strassen(tmpA, B11);
	
	/* P3 */
	tmpB = msum(B12, B22, 1);
	P3 = mprod_strassen(A11, tmpB);

	/* P4 */
	tmpB = msum(B21, B11, 1);
	P4 = mprod_strassen(A22, tmpB);

	/* P5 */
	tmpA = msum(A11, A12, 0);
	P5 = mprod_strassen(tmpA, B22);

	/* P6 */
	tmpA = msum(A21, A11, 1);
	tmpB = msum(B11, B12, 0);
	P6 = mprod_strassen(tmpA, tmpB);

	/* P7 */
	tmpA = msum(A12, A22, 1);
	tmpB = msum(B21, B22, 0);
	P7 = mprod_strassen(tmpA, tmpB);
	
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

	C11 = msum(P1, P4, 0);
	C11 = msum(C11, P5, 1);
	C11 = msum(C11, P7, 0);
	C12 = msum(P3, P5, 0);
	C21 = msum(P2, P4, 0);
	C22 = msum(P1, P3, 0);
	C22 = msum(C22, P2, 1);
	C22 = msum(C22, P6, 0);
	
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

