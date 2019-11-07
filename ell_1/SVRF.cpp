#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

struct StochasticVRFAlgoPara{
    int maxIter;
    double mu;  //2-norm
    double l1Sparsity;
    double eta; //initial step-size
};

void StochasticVRF(const RowMatrix& A,
    const Eigen::VectorXd& label, StochasticVRFAlgoPara& para,
    vector<double>& tDiff, vector<double>& curLoss){
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    vector<int> indices(N);
	for (int i=0; i<N; i++)
		indices[i]=i;
	Eigen::VectorXd x_s(D);
	Eigen::VectorXd w_s(D);
	int base = 4;
    srand(time(0));
	Eigen::SparseMatrix<double, Eigen::ColMajor> colA = A;
	Eigen::SparseMatrix<double, Eigen::ColMajor> colAT = colA.transpose();
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
		base = N/100; //min(1000, base*2);
		int Nt = base-2;
		Eigen::VectorXd Gw = primal_grad_smooth_hinge_loss_reg(
		    colA, colAT, label, w_s, para.mu / N);
		x_s = Eigen::VectorXd(w_s);
		for (int k=0; k<Nt; k++){
			double eta = 10.0/para.l1Sparsity/(k+3);
			int m_k = 100; //min(N, 10*(k+1));
			std::random_shuffle(indices.begin(), indices.end());
			Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
				            A, label, x_s, para.mu / N, m_k, indices);
			Eigen::VectorXd gradw = primal_grad_smooth_hinge_loss_reg(
			                A, label, w_s, para.mu / N, m_k, indices);
			grad = grad - (gradw - Gw);
     		int pos = 0;
	        double value = 0;
			for (int i = 0; i < D; i++){
				if (grad[i] > fabs(value) or grad[i] < -fabs(value)){
				    pos = i;
			        value = grad[i];
			    }
			}
			x_s = (1 - eta) * x_s;
			if (value > 0)
			    x_s[pos] -= eta * para.l1Sparsity;
			else
			    x_s[pos] += eta * para.l1Sparsity;
        	if (k%100==0){
				double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        		cout << curIter<< "-th iter "<<k<<"-th inner loop loss: " << loss << endl;
			}
		}
		w_s = Eigen::VectorXd(x_s);
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
}






