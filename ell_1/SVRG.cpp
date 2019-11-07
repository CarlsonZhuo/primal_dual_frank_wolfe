#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

struct StochasticVRGAlgoPara{
    int maxIter;
    double mu;  //2-norm
    double l1Sparsity;
    double eta; //initial step-size
    bool smallData;
};

void StochasticVRG(const RowMatrix& A,
    const Eigen::VectorXd& label, StochasticVRGAlgoPara& para,
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
    double tSum;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tSum = 0;
		int Nt = min(100, N);
		Eigen::VectorXd Gw = primal_grad_smooth_hinge_loss_reg(
		    colA, colAT, label, w_s, para.mu / N);
		x_s = Eigen::VectorXd(w_s);
		for (int k=0; k<Nt; k++){
			int m_k = min(100, N/10); //min(N, 10*(k+1));
			std::random_shuffle(indices.begin(), indices.end());
			Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
				            A, label, x_s, para.mu / N, m_k, indices);
			Eigen::VectorXd gradw = primal_grad_smooth_hinge_loss_reg(
			                A, label, w_s, para.mu / N, m_k, indices);
			grad = grad - (gradw - Gw);
			tStart = clock();
			x_s = l1_proj(x_s - para.eta * grad, para.l1Sparsity);
			tSum += (double)(clock() - tStart)/CLOCKS_PER_SEC;
			// if (k%100==0){
			// 	double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
   //      		cout << curIter<< "-th iter "<<k<<"-th inner loop loss: " << loss << endl;
			// }
		}
		w_s = Eigen::VectorXd(x_s);
        // tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        tDiff[curIter] = tSum;
        double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
}






