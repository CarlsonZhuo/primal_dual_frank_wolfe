#include "util.h"
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;

struct FrankWolfeAlgoPara{
    int maxIter;
    double mu;  //l1 regularizer
    double l1Sparsity;
    double eta; //initial step-size
};

void FrankWolfe(const SparseMatrix& A,
    const Eigen::VectorXd& label, FrankWolfeAlgoPara& para,
	vector<double>& tDiff, vector<double>& curLoss){
    SparseMatrix AT = A.transpose();
	double ratio = 1.1;
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    Eigen::VectorXd x_i(D);
    x_i.setZero();
    //
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
        ////////////////////// Primal ///////////////////////
        double eta = para.eta * 2.0/(curIter+3);
		Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
				            A, AT, label, x_i, para.mu / N);
		int pos = 0;
		double value = 0;
		for (int i = 0; i < D; i++){
            if (grad[i] > fabs(value) or grad[i] < -fabs(value)){
				pos = i;
			    value = grad[i];
			}
		}
		x_i *= 1-eta;
		if (value>0)
			x_i[pos] -= eta*para.l1Sparsity;
		else
			x_i[pos] += eta*para.l1Sparsity;
     	tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_i, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
    cout << "prediction_accuracy: " << prediction_accuracy(A, label, x_i) << endl;
}



