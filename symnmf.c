#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include "symnmf.h"


/* --- Helper: read input file into double** matrix ------------------------ */
static double **read_input_file(const char *path, int *n, int *d) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Could not open file: %s\n", path);
        exit(1);
    }
    /* First pass: count rows and columns */
    int rows = 0, cols = 0, cur_cols = 0;
    char line[10000];
    while (fgets(line, sizeof(line), fp)) {
        if (rows == 0) {
            /* count commas to determine number of columns */
            for (char *c = line; *c; c++) if (*c == ',') cols++;
            cols++; /* #commas + 1 = #columns */
        }
        rows++;
    }
    rewind(fp);
    double **X = allocate_matrix(rows, cols);
    if (!X) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    /* Second pass: read values */
    int r = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *ptr = strtok(line, ",\n\r");
        int c = 0;
        while (ptr && c < cols) {
            X[r][c] = atof(ptr);
            ptr = strtok(NULL, ",\n\r");
            c++;
        }
        r++;
    }
    fclose(fp);
    *n = rows;
    *d = cols;
    return X;
}

/* --- Helper: print matrix in CSV format ---------------------------------- */
static void print_matrix(double **M, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (j > 0) printf(",");
            printf("%.4f", M[i][j]);
        }
        printf("\n");
    }
}

/* compute A = X * X^T (dot-product similarity) */
void compute_similarity(const double *X, int N, int D, double *A) {
    // initialize
    for (int i = 0; i < N * N; i++) A[i] = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double s = 0.0;
            const double *xi = X + i * D;
            const double *xj = X + j * D;
            for (int t = 0; t < D; t++) s += xi[t] * xj[t];
            A[i * N + j] = s;
            A[j * N + i] = s;
        }
    }
}

/* D is stored as full NxN matrix with zeros off-diagonal and sums on diagonal */
void compute_ddg_from_A(const double *A, int N, double *D) {
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += A[i * N + j];
        for (int j = 0; j < N; j++) D[i * N + j] = 0.0;
        D[i * N + i] = sum;
    }
}

/* compute normalized similarity W = D^{-1/2} * A * D^{-1/2} */
void compute_normalized_similarity(const double *A, int N, double *W) {
    double *D = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) sum += A[i * N + j];
        D[i] = (sum <= 0.0) ? 0.0 : 1.0 / sqrt(sum);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            W[i * N + j] = A[i * N + j] * D[i] * D[j];
        }
    }
    free(D);
}

/* matrix multiply: C = A (n x m) * B (m x p) -> C (n x p) */
static void matmul(const double *A, const double *B, double *C, int n, int m, int p) {
    // initialize
    for (int i = 0; i < n * p; i++) C[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < m; k++) {
            double aik = A[i * m + k];
            for (int j = 0; j < p; j++) {
                C[i * p + j] += aik * B[k * p + j];
            }
        }
    }
}

/* compute frobenius norm squared of (A - B), both n x n */
static double frob_diff_sq(const double *A, const double *B, int n) {
    double s = 0.0;
    for (int i = 0; i < n * n; i++) {
        double d = A[i] - B[i];
        s += d * d;
    }
    return s;
}

/* SymNMF multiplicative updates: minimize ||W - H H^T||_F^2
   Update: H = H .* (W H) ./ (H (H^T H))
*/
void symnmf_solve(const double *W, int N, int k, double *H, double eps, int max_iter) {
    double *WH = malloc(N * k * sizeof(double));
    double *HtH = malloc(k * k * sizeof(double));
    double *H_HtH = malloc(N * k * sizeof(double));
    double *HHt = malloc(N * N * sizeof(double));

    double prev_obj = 0.0;
    // compute initial HHt = H * H^T
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) HHt[i*N + j] = 0.0;
    for (int i = 0; i < N; i++) {
        for (int r = 0; r < k; r++) {
            double hir = H[i*k + r];
            for (int j = 0; j < N; j++) {
                HHt[i*N + j] += hir * H[j*k + r];
            }
        }
    }
    prev_obj = frob_diff_sq(W, HHt, N);

    for (int iter = 0; iter < max_iter; iter++) {
        // WH = W * H
        matmul(W, H, WH, N, N, k);

        // HtH = H^T * H (k x k)
        for (int i = 0; i < k * k; i++) HtH[i] = 0.0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                double s = 0.0;
                for (int t = 0; t < N; t++) s += H[t*k + i] * H[t*k + j];
                HtH[i*k + j] = s;
            }
        }

        // H_HtH = H * HtH  (N x k)
        matmul(H, HtH, H_HtH, N, k, k);

        // update H elementwise: H_ij = H_ij * WH_ij / H_HtH_ij
        for (int i = 0; i < N * k; i++) {
            double denom = H_HtH[i];
            if (denom <= 0) denom = 1e-12; // avoid divide by zero
            double val = H[i] * (WH[i] / denom);
            if (val < 1e-12) val = 1e-12; // keep positive
            H[i] = val;
        }

        // compute HHt = H * H^T
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) HHt[i*N + j] = 0.0;
        for (int i = 0; i < N; i++) {
            for (int r = 0; r < k; r++) {
                double hir = H[i*k + r];
                for (int j = 0; j < N; j++) {
                    HHt[i*N + j] += hir * H[j*k + r];
                }
            }
        }
        double obj = frob_diff_sq(W, HHt, N);
        double diff = fabs(prev_obj - obj);
        if (prev_obj > 0 && diff / prev_obj < eps) break;
        prev_obj = obj;
    }

    free(WH); free(HtH); free(H_HtH); free(HHt);
}

/* --- Main entry point ---------------------------------------------------- */
int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("An Error Has Occurred");
        return 1;
    }
    const char *goal = argv[1];
    const char *filename = argv[2];
    double *X = NULL;
    int N, D;
    if (read_dataset(filename, &X, &N, &D) != 0) {
        printf("An Error Has Occurred");
        return 1;
    }
    double *A = malloc(N * N * sizeof(double));
    compute_similarity(X, N, D, A);
    if (strcmp(goal, "sym") == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.4f", A[i*N + j]);
                if (j < N-1) printf(",");
            }
            printf("");
        }
    } else if (strcmp(goal, "ddg") == 0) {
        double *Dmat = malloc(N * N * sizeof(double));
        compute_ddg_from_A(A, N, Dmat);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.4f", Dmat[i*N + j]);
                if (j < N-1) printf(",");
            }
            printf("");
        }
        free(Dmat);
    } else if (strcmp(goal, "norm") == 0) {
        double *W = malloc(N * N * sizeof(double));
        compute_normalized_similarity(A, N, W);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.4f", W[i*N + j]);
                if (j < N-1) printf(",");
            }
            printf("");
        }
        free(W);
    } else {
        printf("An Error Has Occurred");
        free(A); free(X);
        return 1;
    }
    free(A); free(X);
    return 0;
}