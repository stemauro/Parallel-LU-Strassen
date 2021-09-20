#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>


/* INPUT: A - array of pointers to rows of a square matrix having dimension N
 *        Tol - small tolerance number to detect failure when the matrix is near degenerate
 * OUTPUT: Matrix A is changed, it contains a copy of both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
 *        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1
 *        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N,
 *        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S
 */
int LUPDecompose(double **A, int N, double Tol, int *P) {

    int i, j, k, imax;
    double maxA, *ptr, absA;

    // Permutation matrix initialization
    for (i = 0; i <= N; i++)
        P[i] = i; //Unit permutation matrix, P[N] initialized with N

        // Decomposition
        for (i = 0; i < N; i++) {
            maxA = 0.0;
            imax = i;

            // Check pivot value in next rows
            for (k = i; k < N; k++)
                if ((absA = fabs(A[k][i])) > maxA) {
                    maxA = absA;
                    imax = k;
                }

            if (maxA < Tol) return 0; //failure, matrix is degenerate

            // If current row does not contain pivot value, pivot the matrix
            if (imax != i) {
                //pivoting P
                j = P[i];
                P[i] = P[imax];
                P[imax] = j;

                //pivoting rows of A
                ptr = A[i];
                A[i] = A[imax];
                A[imax] = ptr;

                //counting pivots starting from N (for determinant)
                P[N]++;
            }

            for (j = i + 1; j < N; j++) {
                A[j][i] /= A[i][i];

                for (k = i + 1; k < N; k++)
                    A[j][k] -= A[j][i] * A[i][k];
            }
        }

        return 1;  //decomposition done
}


/* INPUT: A,P filled in LUPDecompose; N - dimension
 * OUTPUT: IA is the inverse of the initial matrix
 */
void LUPInvert(double **A, int *P, int N, double **IA) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            for (int k = 0; k < i; k++)
                IA[i][j] -= A[i][k] * IA[k][j];
        }

        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++)
                IA[i][j] -= A[i][k] * IA[k][j];

            IA[i][j] /= A[i][i];
        }
    }
}


/* INPUT: Matrix, stdout
 */
void print_matrix(double **A, int n){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.4f ", A[i][j]);
        }
        printf("\n");
    }
}


#define threads 2
#define n 10
#define tol 0.000000000001

int main() {

    printf("----- LUP Decomposition and Inversion -----\n\n");

    //    omp_set_num_threads(threads);
    int nthreads = omp_get_max_threads();
    printf("number of threads: %d\n", nthreads);

    int *P=malloc(n*sizeof(int)); // Memory allocation for permutation matrix
    double *A[n]; // Array of rowpointers for A matrix

    // allocate for each row memory for n elements of type double and assign a pointer to the row
    for (int i = 0; i < n; ++i) {
        A[i] = (double *)malloc(n * sizeof(double));
    }

    // A initialization
    double initialization_start, initialization_end, initialization_time;
    initialization_start = omp_get_wtime();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand()%10;
        }
    }
    initialization_end = omp_get_wtime();
    initialization_time = initialization_end - initialization_start;
    printf("Initialization time: %.7f \n", initialization_time);



    printf("Input matrix (A) shape: (%d x %d)\n", n, n);
    //    print_matrix(A, n);


    double decomposition_start, decomposition_end, lup_decomposition_time;
    double inversion_start, inversion_end, lup_inversion_time;

    // LUP decomposition
    decomposition_start = omp_get_wtime();
    int decomposition_flag = LUPDecompose(A, n, tol, P);
    decomposition_end = omp_get_wtime();
    lup_decomposition_time = decomposition_end - decomposition_start;
    printf("\n Decomposition flag: %d \n", decomposition_flag);
    printf("\nLUP decomposition ---\n");
    printf("Decomposition time: %.7f \n", lup_decomposition_time);
    //    printf("\npost decomposition \n");
    //    print_matrix(A, n);


    // Array of rowpointers for Inverse matrix
    double *IA[n];

    // allocate memory for one row and append the rowpointer to the array
    for (int i = 0; i < n; ++i) {
        IA[i] = (double *)malloc(n*sizeof(double));
    }

    // LUP inversion
    if(decomposition_flag){
        inversion_start = omp_get_wtime();
        LUPInvert(A, P, n, IA);
        inversion_end = omp_get_wtime();
        lup_inversion_time = inversion_end - inversion_start;
        printf("\nLUP inversion ---\n");
        printf("Inversion time: %.7f \n", lup_inversion_time);
        //    printf("Post inversion \n");
        //    print_matrix(IA, n);
    }

    return 0;
}








