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


#define n 100
#define tol 0.000000000001

int main() {

    printf("LUP decomposition + inversion\n");
    int *P=malloc(n*sizeof(int));

    /*The point is that it doesn't have any particular magic role,
     * it's just allocating memory just like malloc(), but happens to
     * a) initialize it,and
     * b) have an interface which makes it easy to use when you're going to use the returned pointer as an array.*/
    double A[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = rand()%10;
        }
    }
    double *array_of_rowspointers[n];


    // array initialization
    for (int i = 0; i < n; ++i) {
        array_of_rowspointers[i] = *(A + i);
    }

    // dynamic allocation (calloc or malloc is more efficient)
    /* malloc --> doesn't initialize (faster than calloc), returns a pointer to the beginning of the block
     * calloc --> allocates and initializes memory block to 0.
     * */


    printf("\npre decomposition \n");
    print_matrix(array_of_rowspointers, n);

    int decomposition_flag = LUPDecompose(array_of_rowspointers, n, tol, P);
    printf("\nflag: %d \n", decomposition_flag);

    printf("\npost decomposition \n");
    print_matrix(array_of_rowspointers, n);


    double IA[n][n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            IA[i][j] = rand()%10;
        }
    }
    double *array_of_rowspointersIA[n];
    // array initialization
    for (int i = 0; i < n; ++i) {
        array_of_rowspointersIA[i] = *(IA + i);
    }

    LUPInvert(array_of_rowspointers, P, n, array_of_rowspointersIA);
    printf("\npost inversion \n");
    print_matrix(array_of_rowspointersIA, n);


    return 0;
}
