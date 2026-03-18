#ifndef SYMNMF_H
#define SYMNMF_H


void compute_similarity(const double *X, int N, int D, double *A);
void compute_ddg_from_A(const double *A, int N, double *D);
void compute_normalized_similarity(const double *A, int N, double *W);


/* symnmf_solve: performs multiplicative updates in-place on H (initial H provided)
W: N x N, H: N x k
*/
void symnmf_solve(const double *W, int N, int k, double *H, double eps, int max_iter);


#endif /* SYMNMF_H */