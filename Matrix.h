#ifndef MATRIX_H
#define MATRIX_H
#include <string>
#include <vector>

using namespace std;

class Matrix {
    private:
        /// @brief Number of rows in matrix
        int rows;

        /// @brief Number of columns in 'matrix'
        int cols;

        /// @brief Number of expected floating operations
        int flops;

        /// @brief Number of floating operations done
        int realFlops;

        /// @brief Coefficients matrix
        vector<vector<double>>* matrix;

        /**
         * Normal matrix multiplication by matrix (AX = B)
         * 
         * @param[in] X matrix X
         * @param[in,out] B matrix B
         * @return matrix B result of the multiplication
         */
        vector<vector<double>>* multByMatrix(const vector<vector<double>>* X, vector<vector<double>>* B);

        /**
         * Matrix multiplication by matrix using blocks (AX = B)
         * 
         * @param[in] X matrix X
         * @param[in,out] B matrix B
         * @return matrix B result of the multiplication
         */
        vector<vector<double>>* blockMultByMatrix(const vector<vector<double>>* X, vector<vector<double>>* B);

        /**
         * Row-oriented matrix multiplication by vector (Ax = b)
         * 
         * @param[in] x vector x
         * @param[in,out] b vector b
         * @returns vector b result of the multiplication
         */
        vector<double>* rowMultByVector(const vector<double>* x, vector<double>* b);

        /**
         * Column-oriented matrix multiplication by vector (Ax = b)
         * 
         * @param[in] x vector x
         * @param[in,out] b vector b
         * @returns vector b result of the multiplication
         */
        vector<double>* colMultByVector(const vector<double>* x, vector<double>* b);

        /**
         * Column-oriented linear system solver (Ax = b). Only works with lower triangular coefficients matrix.
         * This is a non-recursive version of the algorithm.
         * 
         * @param[in,out] b vector b
         * @returns solution of the given linear system
         */
        vector<double>* colForwardElimination(vector<double>* b);

        /**
         * Row-oriented linear system solver (Ax = b). Only works with lower triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @returns solution of the given linear system
         */
        vector<double>* rowForwardElimination(vector<double>* b);

        /**
         * Row-oriented linear system solver (Ax = b) using leading-zeros. Only works with lower triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @returns solution of the given linear system
         */
        vector<double>* rowZerosForwardElimination(vector<double>* b);

        /**
         * Non-recursive version for column-oriented linear system solver (Ax = b). Only works with upper triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @returns solution of the given linear system
         */
        vector<double>* colBackwardElimination(vector<double>* b);

        /**
         * Non-recursive version for row-oriented linear system solver (Ax = b). Only works with upper triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @returns solution of the given linear system
         */
        vector<double>* rowBackwardElimination(vector<double>* b);

        /**
         * Matrices blocks multiplication (AX = B)
         * 
         * @param[in] X block matrix X
         * @param[in,out] B block matrix B
         * @param[in] i_start, j_start, k_start start indexes of the current block
         * @param[in] blocksize size of the block
         */
        void blockMultiply(const vector<vector<double>>* X, vector<vector<double>>* B, const int i_start, const int j_start, const int k_start, const int blocksize);

        /**
         * Verify if the matrix is upper triangular
         * 
         * @returns 'true' if matrix is upper triangular, 'false' otherwise.
         */
        bool isUpperTriangular();

        /**
         * Verify if the matrix is lower triangular
         * 
         * @returns 'true' if matrix is lower triangular, 'false' otherwise.
         */
        bool isLowerTriangular();

    public:
        /**
         * Constructor for coefficients matrix
         * 
         * @param[in] rows, cols number of rows and cols of the matrix
         */
        Matrix(const int rows, const int cols);

        /**
         * Constructor for coefficients matrix
         * 
         * @param[in] matrix coefficients matrix
         */
        Matrix(vector<vector<double>>* matrix);

        /// @brief Class destructor
        ~Matrix();

        /**
         * @returns Number of rows in coefficient matrix
         */
        int getRows();

        /**
         * @returns Number of columns in coefficient matrix
         */
        int getCols();
        
        /**
         * @returns Floating operations done
         */
        int getFlops();

        /**
         * Set a value for a given index in coefficients matrix
         * 
         * @param[in] r, c index where the value will be put in
         * @param[in] value value
         */
        void setValue(const int r, const int c, const double value);

        /// @brief Print floating operations done
        void printFlops();

        /// @brief Print the coefficient matrix
        void printMatrix();

        /**
         * Print given matrix A
         * 
         * @param[in] A matrix 
         */
        void printMatrix(vector<vector<double>>* A);

        /**
         * Linear system solver (Ax = b) for lower triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @param[in] mode 'row' (default) for row-oriented solver, or 'col' for column-oriented solver.
         * @returns solution of the given linear system
         */
        vector<double>* forwardElimination(vector<double>* b, const string mode = "row");

        /**
         * Linear system solver (Ax = b) for upper triangular coefficients matrix.
         * 
         * @param[in,out] b vector b
         * @param[in] mode 'row' (default) for row-oriented solver, or 'col' for column-oriented solver.
         * @returns solution of the given linear system
         */
        vector<double>* backwardElimination(vector<double>* b, const string mode = "row");

        /**
         * Matrix multiplication by vector (Ax = b)
         * 
         * @param[in] x vector x
         * @return vector b result of the multiplication
         */
        vector<double>* mult(const vector<double>* x, const string mode = "row");

        /**
         * Matrix multiplication by matrix (AX = B)
         * 
         * @param[in] X matrix X
         * @return matrix B result of the multiplication
         */
        vector<vector<double>>* mult(const vector<vector<double>>* X, const string mode = "row");

        /**
         * Calculates the Cholesky's factor of matrix A and verify if A is positive definite.
         * In a n x n matrix, this algorithm performs n^3/3 floating operations
         * 
         * @returns Cholesky's factor if A is positive definite. Otherwise, returns NULL
         */
        vector<vector<double>>* choleskyFactor();

        /**
         * @returns transpose matrix of A
         */
        vector<vector<double>>* transpose();
};

#endif