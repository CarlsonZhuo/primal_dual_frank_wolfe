#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;



void ProjGrad(const SparseMatrix& A,
    const Eigen::VectorXd& label, AcceleratedProjGradAlgoPara& para,
    vector<double>& tDiff, vector<double>& curLoss){
    SparseMatrix AT = A.transpose();
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    Eigen::VectorXd x_s(D);
    x_s.setZero();
    //
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
        Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
                            A, AT, label, x_s, para.mu / N);
        x_s -= para.eta * grad;
        x_s = l1_proj(x_s, para.l1Sparsity);
        //
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
}





