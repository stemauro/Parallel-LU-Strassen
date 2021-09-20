#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "p-matrix.h"
#include "p-matops.h"

int main(int argc, const char *argv[]) {
	
	int nra = atoi(argv[1]);
	struct fmat_t *mA, *mB, *mC;
	
	/* Space allocation for input matrices */
	mA = f_matrix_calloc(nra, nra);
	mB = f_matrix_calloc(nra, nra);
	
	/* Computations */
	mC = mprod_rowcol(mA, mB);
	
	/* Free matrices */
	f_matrix_free(mC);
	f_matrix_free(mB);
	f_matrix_free(mA);
	return 0;
}
