/**
 * @brief Functions for matrix computations. This code is based on the book Fundamentals of Matrix Computations, by David S. Watkins.
 *
 * @author Rennisson Davi D. Alves
 * Contact: rennisson.alves@gmail.com
 *
 */

#include "Matrix.h"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
using namespace std;

Matrix::Matrix(const int rows, const int cols) {
    this->matrix    = new vector<vector<double>>(rows, vector<double>(cols, 0));
    this->rows      = rows;
    this->cols      = cols;
    this->flops     = 0;
};

Matrix::Matrix(vector<vector<double>>* matrix) {
    this->matrix    = matrix;
    this->rows      = matrix->size();
    this->cols      = matrix[0].size();
    this->flops     = 0;
};

Matrix::~Matrix() { };

int Matrix::getRows() { return rows; }

int Matrix::getCols() { return cols; }

int Matrix::getFlops() { return realFlops; }

void Matrix::setValue(const int r, const int c, const double value) {
    if (r > rows || c > cols || r < 0 || c < 0)
        throw new invalid_argument("Index out of range");
    (*matrix)[r][c] = value;
}


vector<double>* Matrix::mult(const vector<double>* x, const string mode) {
    if (x->size() != cols || x->size() <= 0) throw invalid_argument("Vector invalid for multiplication");

    vector<double>* b = new vector<double>(x->size(), 0);
    realFlops = 0;
    flops = 2*rows*cols;

    // Row-oriented matrix-vector multiply
    if (mode == "row") return rowMultByVector(x, b);
    // Column-oriented matrix-vector multiply
    if (mode == "col") return colMultByVector(x, b);

    cout << "Only allowed row or col-oriented multiplication" << endl;

    delete b;
    return NULL;
}

vector<double>* Matrix::rowMultByVector(const vector<double>* x, vector<double>* b) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            (*b)[i] += (*matrix)[i][j] * (*x)[j];
            realFlops += 2;    
        }
    return b;
}

vector<double>* Matrix::colMultByVector(const vector<double>* x, vector<double>* b) {
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++) {
            (*b)[i] += (*matrix)[i][j] * (*x)[j];
            realFlops += 2;
        }
    return b;
}

vector<vector<double>>* Matrix::mult(const vector<vector<double>>* X, const string mode) {
    if (X->size() <= 0 || X[0].size() <= 0) throw new invalid_argument("Number of rows/cols must be greater than zero");
    if (X->size() != cols) throw new invalid_argument("Number of rows in X must be equals to number of cols in A");

    vector<vector<double>>* B = new vector<vector<double>>(rows, vector<double>(cols, 0));
    realFlops = 0;
    flops = 2 * this->rows * cols * this->cols;

    if (mode == "normal") return multByMatrix(X, B);
    if (mode == "block") return blockMultByMatrix(X, B);
    
    cout << "Only allowed normal or block multiplication" << endl;
    return NULL;
}

vector<vector<double>>* Matrix::multByMatrix(const vector<vector<double>>* X, vector<vector<double>>* B) {
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < this->cols; k++) {
                (*B)[i][j] += (*matrix)[i][k] * (*X)[k][j];
                realFlops += 2;
            }
    return B;
}

vector<vector<double>>* Matrix::blockMultByMatrix(const vector<vector<double>>* X, vector<vector<double>>* B) {
    int blocksize = 3;
    for (int i = 0; i < rows; i += blocksize)
        for (int j = 0; j < X[0].size(); j += blocksize)
            for (int k = 0; k < cols; k += blocksize)
                blockMultiply(X, B, i, j, k, blocksize);
    return B;
}

void Matrix::blockMultiply(const vector<vector<double>>* X, vector<vector<double>>* B,
                            const int i_start, const int j_start, const int k_start, const int blocksize) {
    int xCols = X[0].size();
    for (int i = i_start; i < min(i_start + blocksize, rows); i++)
        for (int j = j_start; j < min(j_start + blocksize, xCols); j++)
            for (int k = k_start; k < min(k_start + blocksize, cols); k++) {
                (*B)[i][j] += (*matrix)[i][k] * (*X)[k][j];
                realFlops += 2;
            }
}

vector<double>* Matrix::forwardElimination(vector<double>* b, const string mode) {
    if (b->size() < 1)          throw invalid_argument("Invalid vector");
    if (!isLowerTriangular())   throw invalid_argument("Matrix is not lower triangular");
    
    realFlops = 0;
    if (mode == "row")          return rowForwardElimination(b);
    if (mode == "row-zeros")    return rowZerosForwardElimination(b);
    if (mode == "col")          return colForwardElimination(b);

    cout << "Only allowed row/col-oriented forward elimination" << endl;
    return NULL;
}

vector<double>* Matrix::rowForwardElimination(vector<double>* b) {
    int size = matrix->size();
    // Row-oriented Forward elimination (Lower triangular coeff. matrix)
    for (int i = 0; i < size; i++) {
        // Foolproof verification (g can't be singular) 
        if ((*matrix)[i][i] == 0) throw invalid_argument("L is singular");

        for (int j = 0; j < i; j++) {
            (*b)[i] -= (*matrix)[i][j] * (*b)[j];
            realFlops += 2;
        }

        (*b)[i] /= (*matrix)[i][i];
    }
    flops = size * (size - 1);
    return b;
}

vector<double>* Matrix::rowZerosForwardElimination(vector<double>* b) {
    int size = matrix->size();
    // Row-oriented Forward elimination using leading-zeros (Lower triangular coeff. matrix)
    int k = 0;
    for (int i = 0; i < size; i++) {
        if ((*b)[k] != 0) break;
        k++;
    }

    for (int i = k; i < size; i++) {
        // Foolproof verification (g can't be singular) 
        if ((*matrix)[i][i] == 0) throw invalid_argument("L is singular");

        for (int j = k; j < i; j++) {
            (*b)[i] -= (*matrix)[i][j] * (*b)[j];
            realFlops += 2;
        }

        (*b)[i] /= (*matrix)[i][i];
    }
    flops = (size - k) * (size - k);
    return b;
}

vector<double>* Matrix::colForwardElimination(vector<double>* b) {
    int size = matrix->size();
    // Column-oriented Forward elimination (Nonrecursive function) (Lower triangular coeff. matrix)
    for (int j = 0; j < size; j++) {
        if ((*matrix)[j][j] == 0) throw invalid_argument("L is singular");

        (*b)[j] /= (*matrix)[j][j];

        for (int i = j+1; i < size; i++) {
            (*b)[i] -= (*matrix)[i][j] * (*b)[j];
            realFlops += 2;
        }
    }
    flops = size * (size - 1);
    return b;
}

// Upper triangular matrix only
vector<double>* Matrix::backwardElimination(vector<double>* b, const string mode) {
    if (b->size() < 1)          throw invalid_argument("Invalid vector");
    if (!isUpperTriangular())   throw invalid_argument("Matrix is not upper triangular");
    
    realFlops = 0;
    if (mode == "row") return rowBackwardElimination(b);
    if (mode == "col") return colBackwardElimination(b);

    cout << "Only allowed row/col-oriented forward elimination" << endl;
    return NULL;
}

vector<double>* Matrix::rowBackwardElimination(vector<double>* b) {
    int size = matrix->size();

    // Row-oriented Backward elimination (Upper triangular coeff. matrix)
    for (int i = size - 1; i >= 0; i--) {
        // Foolproof verification (g can't be singular) 
        if ((*matrix)[i][i] == 0) throw invalid_argument("U is singular");
        
        for (int j = size - 1; j >= i + 1; j--) {
            (*b)[i] -= (*matrix)[i][j] * (*b)[j];
            realFlops += 2;
        }

        (*b)[i] /= (*matrix)[i][i];
    }
    flops = size * (size - 1);
    return b;
}

vector<double>* Matrix::colBackwardElimination(vector<double>* b) {
    int size = matrix->size();

    for (int j = size - 1; j >= 0; j--) {
        // Verificação de singularidade
        if ((*matrix)[j][j] == 0) {
            throw invalid_argument("U is singular");
        }

        (*b)[j] /= (*matrix)[j][j];

        for (int i = 0; i < j; i++) {
            (*b)[i] -= (*matrix)[i][j] * (*b)[j];
            realFlops += 2;
        }
    }

    flops = size * (size - 1);
    return b;
}

vector<vector<double>>* Matrix::choleskyFactor(string mode) {
    if (rows != cols) {
        cout << "Matriz not square" << endl;
        return nullptr;
    }

    if (mode == "inner") return innerProdCholeskyFactor();

    if (mode == "outer") return outerProdCholeskyFactor(vector<vector<double>> ((*this->matrix)), new vector<vector<double>>(rows, vector<double>(cols, 0)), 0, 0);

    cout << "Only allowed inner/outer product form" << endl;
    return nullptr;
}

/// Inner-product form of Cholesky decomposition 
vector<vector<double>>* Matrix::innerProdCholeskyFactor() {
    // Copy 'matrix' into 'R'
    vector<vector<double>>* R = new vector<vector<double>>(*matrix);

    for (int i = 0; i < rows; i++) {
        for (int k = 0; k < i; k++) (*R)[i][i] -= (*R)[k][i] * (*R)[k][i];
        if ((*R)[i][i] < 0) throw invalid_argument("Matrix is not positive definite"); 
        (*R)[i][i] = sqrt((*R)[i][i]);

        for (int j = i + 1; j < rows; j++) {
            for (int k = 0; k < i; k++) (*R)[i][j] -= (*R)[k][i] * (*R)[k][j];
            (*R)[i][j] /= (*R)[i][i];
        }

        // Since the Cholesky's algorithm changes only the upper part of A, this loop makes the lower part zero
        for (int j = 0; j < i; j++) (*R)[i][j] = 0;
    }

    /// Here's a version without creating any copy of original matrix
    // vector<vector<double>>* R = new vector<vector<double>>(rows, vector<double>(cols, 0.0));

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j <= i; j++) {
    //         double sum = 0.0;
    //         if (i == j) {
    //             for (int k = 0; k < j; k++) {
    //                 sum += (*R)[k][j] * (*R)[k][j];
    //             }
    //             (*R)[j][j] = sqrt(matrix[j][j] - sum);
    //             if ((*R)[j][j] <= 0) {
    //                 throw invalid_argument("Matrix is not positive definite");
    //             }
    //         } else {
    //             for (int k = 0; k < j; k++) {
    //                 sum += (*R)[k][i] * (*R)[k][j];
    //             }
    //             (*R)[j][i] = (matrix[j][i] - sum) / (*R)[j][j];
    //         }
    //     }
    // }

    return R;
}

vector<vector<double>>* Matrix::outerProdCholeskyFactor(vector<vector<double>> A, vector<vector<double>>* R, const int row, const int col) {
    // Basis case: 0x0 Matrix, a trivial case   
    if (row == rows || col == cols) return R;

    // r[1][1] = sqrt(a[1][1])
    (*R)[row][col] = sqrt(A[row][col]);
    vector<double>* S = new vector<double>();
    for (int i = col+1; i < (*R)[0].size(); i++) {
        S->push_back((1 / (*R)[row][col]) * A[row][i]);  // s^T = r[1][1]^(-1) * b^T
        (*R)[row][i] = S->back();  // Also put the values in R
    }

    // Outer product. That's why this method is called Outer Product Form of Cholesky's Method
    vector<vector<double>>* SS = outerProduct(S, transpose(S));

    for (int i = 0; i < SS->size(); i++)
    for (int j = 0; j < SS[0].size(); j++)
        // Fixing the indexes so we can use the same matrix through all recursive calls.
        A[row + i + 1][col + j + 1] -= (*SS)[i][j];

    // Deallocating memory
    delete S; 
    delete SS;

    outerProdCholeskyFactor(A, R, row+1, col+1);

    return R;
}

vector<vector<double>>* Matrix::transpose() {
    vector<vector<double>>* transpose = new vector<vector<double>>(rows, vector<double>(cols, 0.0));

    for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
    (*transpose)[j][i] = (*matrix)[i][j];
    return transpose;
}

vector<double>* Matrix::transpose(vector<double>* X) {
    vector<double>* transpose = new vector<double>();

    for (int i = 0; i < X->size(); i++) transpose->push_back((*X)[i]);
    return transpose;
}

vector<vector<double>>* Matrix::outerProduct(vector<double>* u, vector<double>* v) {
    int rows = u->size();
    int cols = v[0].size();
    vector<vector<double>>* P = new vector<vector<double>>(rows, vector<double>(cols, 0));
    
    for (int i = 0; i < rows; i++) 
    for (int j = 0; j < cols; j++)
    (*P)[i][j] = (*u)[i] * (*v)[j];

    return P;
}

// PRINT FUNCTIONS
void Matrix::printFlops() {
    cout << "Expected flops: " << flops << endl;
    cout << "Flops counted:  " << realFlops << endl;
}

void Matrix::print() {
    cout << "Matrix:" << endl << "[";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) cout << "\t" << (*matrix)[i][j];
        if (i == rows - 1) cout << " ]";
        cout << endl;
    }
}

void Matrix::print(const vector<double>* v) {
    cout << "[ ";
    for (double value : (*v)) cout << value << " ";
    cout << "]" << endl;
}

void Matrix::print(const vector<vector<double>>* A) {
    int rows = A->size();
    int cols = A[0].size();

    cout << "[";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) cout << "\t" << (*A)[i][j];
        if (i == rows - 1) cout << " ]";
        cout << endl;
    }
}


bool Matrix::isLowerTriangular() {
    // If matrix is not squared, then cannot be triangular
    if (rows != cols) return false;

    for (int i = 0; i < rows; i++)
    for (int j = i+1; j < cols; j++)
    if ((*matrix)[i][j] != 0) return false;

    cout << "Lower triangular matrix" << endl;
    return true;
}

bool Matrix::isUpperTriangular() {
    // If matrix is not squared, then cannot be triangular
    if (rows != cols) return false;

    for (int i = 0; i < rows; i++)
    for (int j = 0; j < i; j++)
    if ((*matrix)[i][j] != 0) return false;

    cout << "Upper triangular matrix" << endl;
    return true;
}

double Matrix::getValue(int row, int col) {
    if (row < 0 || col < 0 || row >= rows || col >= cols) throw invalid_argument("Index out of range");
    return (*matrix)[row][col];
}