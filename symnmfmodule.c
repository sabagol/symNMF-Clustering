#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "symnmf.h"
#include <stdlib.h>
#include <stdio.h>

/* helpers to convert between numpy float32 arrays and C double arrays */
static double* array_to_c(PyArrayObject* arr, int *rows, int *cols) {
    if (PyArray_TYPE((PyArrayObject*)arr) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "Array must be float32");
        return NULL;
    }
    if (PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be 2-D");
        return NULL;
    }
    *rows = (int)PyArray_DIM(arr, 0);
    *cols = (int)PyArray_DIM(arr, 1);
    double *c_arr = (double*)calloc((*rows) * (*cols), sizeof(double));
    float *data = (float*)PyArray_DATA(arr);
    for (int i = 0; i < (*rows) * (*cols); i++) c_arr[i] = (double)data[i];
    return c_arr;
}

static PyObject* c_to_array(const double* data, int rows, int cols) {
    npy_intp dims[2] = {rows, cols};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!arr) return NULL;
    float* arr_data = (float*)PyArray_DATA((PyArrayObject*)arr);
    for (int i = 0; i < rows*cols; i++) arr_data[i] = (float)data[i];
    return arr;
}

static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyArrayObject* X_py;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_py)) return NULL;
    int N, D;
    double *X = array_to_c(X_py, &N, &D);
    if (!X) return NULL;
    double *A = (double*)calloc(N*N, sizeof(double));
    compute_similarity(X, N, D, A);
    PyObject* out = c_to_array(A, N, N);
    free(X); free(A);
    return out;
}

static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyArrayObject* X_py;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_py)) return NULL;
    int N, D;
    double *X = array_to_c(X_py, &N, &D);
    if (!X) return NULL;
    double *A = (double*)calloc(N*N, sizeof(double));
    double *Dmat = (double*)calloc(N*N, sizeof(double));
    compute_similarity(X, N, D, A);
    compute_ddg_from_A(A, N, Dmat);
    PyObject* out = c_to_array(Dmat, N, N);
    free(X); free(A); free(Dmat);
    return out;
}

static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyArrayObject* X_py;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X_py)) return NULL;
    int N, D;
    double *X = array_to_c(X_py, &N, &D);
    if (!X) return NULL;
    double *A = (double*)calloc(N*N, sizeof(double));
    double *W = (double*)calloc(N*N, sizeof(double));
    compute_similarity(X, N, D, A);
    compute_normalized_similarity(A, N, W);
    PyObject* out = c_to_array(W, N, N);
    free(X); free(A); free(W);
    return out;
}

/* symnmf(W, H_init, eps, max_iter) --> returns H_final
   W: float32 NxN, H_init: float32 Nxk
*/
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyArrayObject *W_py, *H_py;
    double eps = 1e-4;
    int max_iter = 300;
    if (!PyArg_ParseTuple(args, "O!O!|di", &PyArray_Type, &W_py, &PyArray_Type, &H_py, &eps, &max_iter)) return NULL;
    int N1, N2, N3, k;
    double *W = array_to_c(W_py, &N1, &N2);
    if (!W) return NULL;
    double *H = array_to_c(H_py, &N3, &k);
    if (!H) { free(W); return NULL; }
    if (N1 != N2 || N1 != N3) {
        free(W); free(H);
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch between W and H");
        return NULL;
    }
    symnmf_solve(W, N1, k, H, eps, max_iter);
    PyObject* out = c_to_array(H, N1, k);
    free(W); free(H);
    return out;
}

static PyMethodDef SymNMFMethods[] = {
    {"sym", py_sym, METH_VARARGS, "Compute similarity matrix"},
    {"ddg", py_ddg, METH_VARARGS, "Compute diagonal degree matrix"},
    {"norm", py_norm, METH_VARARGS, "Compute normalized similarity matrix"},
    {"symnmf", py_symnmf, METH_VARARGS, "Perform SymNMF (W, H_init, eps=1e-4, max_iter=300)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",
    NULL,
    -1,
    SymNMFMethods
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (m == NULL) return NULL;
    import_array();
    return m;
}