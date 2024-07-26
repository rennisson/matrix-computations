#include "Matrix.h"
#include <iostream>

using namespace std;

int main() {
    Matrix* matrix = new Matrix(4, 4);
    matrix->setValue(0, 0, 4);
    matrix->setValue(0, 1, -2);
    matrix->setValue(0, 2, 4);
    matrix->setValue(0, 3, 2);
    matrix->setValue(1, 0, -2);
    matrix->setValue(1, 1, 10);
    matrix->setValue(1, 2, -2);
    matrix->setValue(1, 3, -7);
    matrix->setValue(2, 0, 4);
    matrix->setValue(2, 1, -2);
    matrix->setValue(2, 2, 8);
    matrix->setValue(2, 3, 4);
    matrix->setValue(3, 0, 2);
    matrix->setValue(3, 1, -7);
    matrix->setValue(3, 2, 4);
    matrix->setValue(3, 3, 7);

    //int value = 9;
    //vector<double>* X = new vector<double>({0.6667, 3.776, 0.75});

    //vector<vector<double>>::iterator row;
    //vector<double>::iterator col;

    // for (row = X->begin(); row != X->end(); row++) {
    //     for (col = row->begin(); col != row->end(); col++)
    //         *col = ++value;
    // }
    
    // try {
    //     string mode = "col";
    //     vector<double>* result = matrix->backwardElimination(X, mode);

    //     cout << "X: [";
    //     for (vector<double>::iterator col = X->begin(); col != X->end(); ++col) {
    //         cout << "\t" << *col;
    //     }
    //     cout << "\t]" << endl;
    //     matrix->printFlops();

    // } catch (const exception& e) {
    //     cerr << "Error: " << e.what() << endl;
    // }

    vector<vector<double>>* R = matrix->choleskyFactor();
    matrix->printMatrix(R);
    matrix->printMatrix();
    matrix->printMatrix(matrix->transpose());

    delete matrix;
    return 0;
}