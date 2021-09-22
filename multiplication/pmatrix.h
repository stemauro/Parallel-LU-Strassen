#ifndef MAX
#define MAX(a, b) ( ((a) > (b)) ? (a) : (b) )
#endif

struct fblock_t {
	size_t size;
	float *data;
};

struct fmat_t {
	size_t size1;
	size_t size2;
	size_t tda;
	float *data;
	struct fblock_t *block;
	int owner;
};

struct fblock_t *f_block_alloc(size_t n);
void f_block_free(struct fblock_t *b);
struct fmat_t *f_matrix_alloc(size_t n1, size_t n2);
void f_matrix_free(struct fmat_t *m);
void f_matrix_insert(struct fmat_t *m, int ix, int iy, float val);
void f_matrix_init(struct fmat_t *m, float *v, size_t len, int n_threads);
struct fmat_t *f_matrix_calloc(size_t n1, size_t n2, int n_threads);
struct fmat_t *f_matrix_submatrix(struct fmat_t *m,
		        size_t k1, size_t k2,
		        size_t n1, size_t n2);
struct fmat_t *f_matrix_join(struct fmat_t *m1,
			     struct fmat_t *m2,
			     struct fmat_t *m3,
			     struct fmat_t *m4);
void f_matrix_pprint(struct fmat_t *m);
float *f_matrix_randr(size_t n, size_t l, size_t u);

/***************************************************************************
 * Source code
 * ************************************************************************/

struct fblock_t *f_block_alloc(size_t n) {

        struct fblock_t *b;

        b = malloc(sizeof(struct fblock_t));
        if (b == NULL) {return NULL;}
        b->size = n;
        b->data = malloc(n * sizeof(float));

        if ((b->data == NULL)) {
                free(b);
                return NULL;
        }

        return b;
}


void f_block_free(struct fblock_t *b) {

        if (b != NULL) {
                free(b->data);
                free(b);
        }
}


struct fmat_t *f_matrix_alloc(size_t n1, size_t n2) {


        struct fblock_t *b = f_block_alloc(n1 * n2);
        if (b == NULL) {return NULL;}


        struct fmat_t *m = malloc(sizeof(struct fmat_t));
        if (m == NULL) {
                printf("Bad memory allocation.\n");
                f_block_free(b);
                return NULL;
        }

        m->size1 = n1;
        m->size2 = n2;
        m->tda = n2;
        m->data = b->data;
        m->block = b;
        m->owner = 1;

        return m;
}


void f_matrix_free(struct fmat_t *m) {

        if (m != NULL) {
                if (m->owner != 0) {
                        f_block_free(m->block);
                }
                free(m);
        }
}


void f_matrix_insert(struct fmat_t *m, int ix, int iy, float val) {
	
	/* Safe
        if ((ix < 0) || (ix > (m->size1 - 1))) {
                printf("Row index is out of bound.\n");
        }
        else if ((iy < 0) || (iy > (m->size2 - 1))) {
                printf("Column index is out of bound.\n");
        } else {
                float *dd;
                dd = m->data;
                dd[ix * m->tda + iy] = val;
        }
	*/

	/* Unsafe */
	(m->data)[ix * m->tda + iy] = val;
}

void f_matrix_init(struct fmat_t *m, float *v, size_t len, int n_threads) {
	
	if (len != (m->size1 * m->size2)) {
		printf("ERROR: Bad initialization.\n");
		printf("Input cardinality doesn't match matrix dimensions.\n");
		exit(1);
	} else {
		int i, j;
		int chunks = MAX( 1, (int)(ceil(len / n_threads)) );
		#pragma omp parallel for collapse(2) \
		private(i, j) schedule(static, chunks)
		for (i = 0; i < m->size1; i++) {
			for (j = 0; j < m->size2; j++) {
				f_matrix_insert(m, i, j, v[i*m->tda+j]);
			}
		}
	}
}	


struct fmat_t *f_matrix_calloc(size_t n1, size_t n2, int n_threads) {
	
	struct fmat_t *m;
	m = f_matrix_alloc(n1, n2);

	if (m == NULL) {return NULL;}
	
	int dim = m->block->size;
	int chunks = MAX( 1, (int)(ceil(dim / n_threads)) );
	int i;

	#pragma omp parallel for \
	private(i) shared(m, dim, chunks) \
       	schedule(static, chunks)	
	for (i = 0; i < dim; i++){
		(m->data)[i] = 0.0;
	}

	return m;
}


struct fmat_t *f_matrix_submatrix(struct fmat_t *m,
			size_t k1, size_t k2,
			size_t n1, size_t n2) {
	
	if (k1 >= m->size1) {
		printf("ERROR. Row index is out of range.\n");
		return NULL;
	}
	else if (k1 >= m->size2) {
		printf("ERROR. Column index is out of range.\n");
		return NULL;
	}
	else if ((k1 + n1) > m->size1) {
		printf("ERROR. First dimension overflows matrix.\n");
		return NULL;
	}
	else if ((k2 + n2) > m->size2) {
		printf("ERROR. Second dimension overflows matrix.\n");
		return NULL;
	}
	
	struct fmat_t *s;
	s = (struct fmat_t *) malloc (sizeof(struct fmat_t));
	if (s == NULL) {return NULL;}

	s->size1 = n1;
	s->size2 = n2;
	s->tda = m->tda;
	s->data = m->data + (k1 * m->tda + k2);
	s->block = m->block;
	s->owner = 0;

	return s;	
}


struct fmat_t *f_matrix_join(struct fmat_t *m1,
		             struct fmat_t *m2,
			     struct fmat_t *m3,
			     struct fmat_t *m4) {
	
	if ((m1 == NULL) || (m2 == NULL) ||
	    (m3 == NULL) || (m4 == NULL)) {
		return NULL;
	}

	size_t nrc = m1->size2;	
	struct fmat_t *m;

	m = f_matrix_alloc(2*nrc, 2*nrc);
	float *dd = m->data;
	if (m == NULL) {return NULL;}
	
	int i, j;
	#pragma omp parallel sections \
	private(i, j) shared(m, m1, m2, m3, m4)
	{
	
	#pragma omp section
	{
	for (i = 0; i < nrc; i++) {
		for (j = 0; j < nrc; j++) {
			//f_matrix_insert(m, i, j, (m1->data)[i * nrc + j]);
			dd[i*m->tda+j] = (m1->data)[i*nrc+j];
		}
	}
	}
	
	#pragma omp section
	{
	for (i = 0; i < nrc; i++) {
		 for (j = 0; j < nrc; j++) {
			 //f_matrix_insert(m, i, j + nrc, (m2->data)[i * nrc + j]);
			 dd[i*m->tda+(j+nrc)] = (m2->data)[i*nrc+j];
		 }
	}
	}

	#pragma omp section
	{
	for (i = 0; i < nrc; i++) {
		for (j = 0; j < nrc; j++) {
			//f_matrix_insert(m, i + nrc, j, (m3->data)[i * nrc + j]);
			dd[(i+nrc)*m->tda+j] = (m3->data)[i*nrc+j];
		}
	}
	}
	
	#pragma omp section
	{
	for (i = 0; i < nrc; i++) {
		for (j = 0; j < nrc; j++) {
			//f_matrix_insert(m, i + nrc, j + nrc, (m4->data)[i * nrc + j]);
			dd[(i+nrc)*m->tda+(j+nrc)] = (m4->data)[i*nrc+j];
		}
	}		 		
	}
	}
	return m;
}


void f_matrix_pprint(struct fmat_t *m) {

        if (m != NULL) {
                float *dd = m->data;
                printf("([\n");
                for (int i = 0; i < m->size1; i++) {
			printf("  [");
                        for (int j = 0; j < m->size2; j++) {
                                printf("%10.4f\t", dd[i * m->tda + j]);
                        }
                        printf("]\n");
                }
                printf("];\n");
                printf("shape: (%ld,%ld))\n", m->size1, m->size2);
        }
}


float *f_matrix_randr(size_t n, size_t l, size_t u) {
	
	float *v;
	v = (float *)malloc(n * sizeof(*v));
	if (v == NULL) {return NULL;}
	
	srand(time(NULL));
	for (int i = 0; i < n; i++)
		v[i] = l + (float)rand() / (float)(RAND_MAX / u);

	return v;
}
