#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include "matops.h"

int main(int argc, const char *argv[]) {

	struct fmat_t *mA, *mB;
	struct fmat_t *sA, *sB, *sCn, *sCs;
	int nra = atoi(argv[1]);

	mA = f_matrix_alloc(nra, nra);
	mB = f_matrix_alloc(nra, nra);
	
	
	float *vA, *vB;
	vA = f_matrix_randr(nra * nra, 0, 10);
	vB = f_matrix_randr(nra * nra, 0, 10);
	
	/*
	float vA[] = {1, 8, 4, 5,
		      2, 9, 4, 1,
		      8, 7, 3, 0,
		      2, 2, 2, 2};

	float vB[] = {3, 7, 4, 8,
		      6, 6, 6, 6,
		      1, 4, 4, 0,
		      3, 2, 2, 0};
	*/

	printf("\nInitilizing matrices...\n");
	f_matrix_init(mA, vA, nra * nra);
	f_matrix_init(mB, vB, nra * nra);

	//f_matrix_pprint(mA);
	//f_matrix_pprint(mB);

	/* Computations */
	clock_t start, end;

	printf("\nPerforming naive multiplication...\n");
	start = clock();
	sCn = mprod_rowcol(mA, mB);
	end = clock();
	printf("Finished in %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
	
	printf("\nPerforming Strassen multiplication...\n");
	start = clock();
	sCs = mprod_strassen(mA, mB);
	end = clock();
	printf("Finished in %fs\n", (double)(end - start) / CLOCKS_PER_SEC);
	//f_matrix_pprint(sCs);
	//f_matrix_pprint(sCn);

	/* Free matrices */
	free(sCs);
	free(sCn);
	free(vB);
	free(vA);
	f_matrix_free(mB);
	f_matrix_free(mA);
	return 0;
}
