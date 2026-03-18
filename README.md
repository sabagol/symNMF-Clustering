Symmetric Non-negative Matrix Factorization (symNMF) Clustering
Project Overview
This repository contains a high-performance implementation of the Symmetric Non-negative Matrix Factorization (symNMF) clustering algorithm. The project is designed as a hybrid system, leveraging ANSI C for computationally intensive matrix operations and Python for data processing, initialization, and comparative analysis.
The implementation includes a custom Python C API wrapper, allowing for seamless, low-overhead integration between high-level analysis and high-performance low-level execution.
Key Features
Performance Engineering: Optimized matrix multiplication and update rules implemented in C to handle multi-dimensional datasets efficiently.
C/Python Integration: Custom C extension module (symnmfmodule.c) using the Python C API for high-speed data transfer between layers.
Algorithmic Rigor: Complete mathematical pipeline including the computation of Similarity Matrices, Diagonal Degree Matrices, and Normalized Graph Laplacians.
Benchmarking Suite: Comparative analysis tool (analysis.py) that evaluates symNMF against K-Means using Silhouette Scores from scikit-learn.
Memory Safety: Rigorous memory management in C, verified with Valgrind to ensure no leaks or corruption during heavy iterations.
Technical Stack
Languages: C (std=c99), Python 3.
Libraries: NumPy (used for initialization and C-API data handling), Scikit-Learn (for metrics).
Build Tools: Makefile (for C executable), Setuptools (for Python extension).
Mathematical Foundation
The algorithm seeks to find a non-negative matrix  that minimizes the squared Frobenius norm:
Where  is the normalized similarity matrix derived from the input data . The update rule is implemented using a multiplicative approach:
File Structure
symnmf.c / symnmf.h: Core C implementation and header.
symnmfmodule.c: Python C API wrapper.
symnmf.py: Python interface and H-matrix initialization.
analysis.py: Benchmarking and Silhouette score comparison.
setup.py: Build script for the C extension.
Makefile: Script for building the standalone C executable.
Getting Started
Build the Python extension:
python3 setup.py build_ext --inplace


Run the analysis:
python3 analysis.py <k_clusters> <data_file.txt>


Author
Saba Golbandi
