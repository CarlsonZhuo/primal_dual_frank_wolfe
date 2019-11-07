#include "util.h"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <time.h>
#include <cstdlib>
using Eigen::MatrixXd;

using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;
typedef Eigen::Triplet<double> T;

int main(){
    int rows = 10;
    int cols = 10;
    int l0Spar = 3;
    double l1Spar = 3;

    srand((unsigned int) time(0));
    Eigen::VectorXd w(10);
    SparseMatrix X(rows, cols);
    w.setRandom();
    w = l0_proj(w, l0Spar);
    w = l1_proj(w, l1Spar);
    // y = X * w;

    cout << w << endl;

    sparseMatGen(X);
    cout << X << endl;

    Eigen::VectorXd y = X * w;
    
    cout << y << endl;
    for(int i = 0; i < y.size(); i ++){
        if (y[i] >= 0)
            y[i] = 1;
        else
            y[i] = -1;
    }

    cout << y << endl;
}
