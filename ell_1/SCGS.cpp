#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <algorithm>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

struct StochasticCGSAlgoPara{
    int maxIter;
    double mu;  //2-norm
    double l1Sparsity;
    double eta; //initial step-size
	double L;
};

Eigen::VectorXd CndG(const Eigen::VectorXd& G, const Eigen::VectorXd& Q, double beta, double eta, double tau);

void StochasticCGS(const RowMatrix& A,
    const Eigen::VectorXd& label, StochasticCGSAlgoPara& para,
    vector<double>& tDiff, vector<double>& curLoss){
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    vector<int> indices(N);
	for (int i=0; i<N; i++)
		indices[i]=i;
	Eigen::VectorXd z_s = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd q_s=z_s;
	Eigen::VectorXd w_s=z_s;
	srand(time(0));
	Eigen::SparseMatrix<double, Eigen::ColMajor> colA = A;
	Eigen::SparseMatrix<double, Eigen::ColMajor> colAT = colA.transpose();
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
		double gamma = 3.0/(curIter+3);
		z_s = (1-gamma) * w_s + gamma * q_s;
		int m_k = min(N, max(100, N/20));
		std::random_shuffle(indices.begin(), indices.end());
		Eigen::VectorXd Gz = primal_grad_smooth_hinge_loss_reg(
		    colA, label, z_s, para.mu / N, m_k, indices);
		/*if (curIter>10 and curLoss[curIter-2]<curLoss[curIter-1]){
			q_s = CndG(Gz, q_s, 4.0*para.L/(curIter+3), para.L*pow(para.l1Sparsity,2)/(1+curIter)/(2+curIter)/D, para.l1Sparsity);
		}
		else*/
		q_s = q_s - (curIter+1)/3.0/para.L * Gz;
		q_s = l1_proj(q_s, para.l1Sparsity);
		w_s = (1-gamma) * w_s + gamma * q_s;
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, w_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        curLoss[curIter] = loss;
    }
}


Eigen::VectorXd CndG(const Eigen::VectorXd& G, const Eigen::VectorXd& Q, double beta, double eta, double tau){
	Eigen::VectorXd W=Q;
	int iter=0;
	while (1){
		iter++;
		Eigen::VectorXd g = G+beta*(W-Q);
		int maxI, minI;
		double maxV = G.maxCoeff(&maxI);
		double minV = G.minCoeff(&minI);
		Eigen::VectorXd D = W; //Eigen::VectorXD::Zero(D);
		if (maxV>-minV){
			D[maxI] += tau;
		}
		else{
			maxV = -minV;
			maxI = minI;
			D[maxI] -= tau;
		} // D = W - V
		double gap = g.dot(D);// + maxV*tau;
		cout<<"iter: "<<iter<<", gap:"<<gap<<", eta:"<<eta<<endl;
		if (gap<=eta){
			return W;
		}
		double a = max(0., min(1.0, gap/(beta*pow(D.norm(),2))));
		W = W - a*D; //W+a(V-W)
	}

}




