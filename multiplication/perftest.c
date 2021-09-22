#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include "pmatrix.h"
#include "pmatops.h"

#define REPS (5)
int main(int argc, char *argv[]) {
	
	size_t dim = atoi(argv[1]);	
	int n_threads = atoi(argv[2]);
	struct fmat_t *A, *B, *C;

	float *vA;	
	vA = f_matrix_randr(dim * dim, 0, 5);
	//float vA[N*N] = {11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44};
	
	/*float vA[N*N];
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			//vA[i * N + j] = 10.0 * (i + 1) + j;
			vA[i*N+j] = 10.0 * (i+1) + j + 1;
		}
	}
	*/

		
	A = f_matrix_alloc(dim,dim);
	//B = f_matrix_alloc(N, N);	
	
	f_matrix_init(A, vA, dim * dim, n_threads);
	//f_matrix_pprint(A);
	

	double start, end;
	for (int r = 0; r < REPS; r++) {
		start = omp_get_wtime();
		//C = mprod_strassen(A, A, n_threads);
		C = mprod_task(A, A, n_threads, 1);
		//C = mprod_rowcol(A, A, n_threads);	
		end = omp_get_wtime();

		printf("dim:%ld concurrency:%d time:%f\n", dim, n_threads, end - start);
	}
	//f_matrix_pprint(C);

	f_matrix_free(C);
	//f_matrix_free(B);
	f_matrix_free(A);	


	free(vA);	
	return 0;

}



