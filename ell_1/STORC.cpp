#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

struct StochasticTORCAlgoPara{
    int maxIter;
    double mu;  //2-norm
    double l1Sparsity;
    double L; //Lipschitz constant
    bool smallData;
};

void StochasticTORC(const RowMatrix& A,
    const Eigen::VectorXd& label, StochasticTORCAlgoPara& para,
    vector<double>& tDiff, vector<double>& curLoss){
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    vector<int> indices(N);
	for (int i=0; i<N; i++)
		indices[i]=i;
	Eigen::VectorXd x_s = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd w_s(x_s);
	srand(time(0));
	Eigen::SparseMatrix<double, Eigen::ColMajor> colA = A;
	Eigen::SparseMatrix<double, Eigen::ColMajor> colAT = colA.transpose();
	//
    clock_t tStart;
	int base = 8;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
		int Nt = min(min(N/20,200), base*2-2);
		if (para.smallData)
			Nt = min(min(N/5, 200), base*2-2);
		base *= 2;
		Eigen::VectorXd y_s(w_s);
		Eigen::VectorXd Gy = primal_grad_smooth_hinge_loss_reg(
		    colA, colAT, label, y_s, para.mu / N);
		x_s = Eigen::VectorXd(y_s);
		for (int k=0; k<Nt; k++){
			double beta = 3.0*para.L/(k+1);
			double gamma = 2.0/(k+2);
			Eigen::VectorXd z_s = (1-gamma)*y_s + gamma *x_s;
			int m_k = min(N, 5*(k+1));
			std::random_shuffle(indices.begin(), indices.end());
			Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
				            A, label, z_s, para.mu / N, m_k, indices);
			Eigen::VectorXd grady = primal_grad_smooth_hinge_loss_reg(
			                A, label, w_s, para.mu / N, m_k, indices);
			grad = grad - (grady - Gy);
			x_s = l1_proj(-grad/beta + x_s, para.l1Sparsity);
			y_s = (1-gamma)*y_s+gamma*x_s;
     	}
		w_s = Eigen::VectorXd(x_s);
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
}




