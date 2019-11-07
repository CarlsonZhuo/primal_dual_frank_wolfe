#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <math.h>
using Eigen::MatrixXd;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;

struct AcceleratedProjGradAlgoPara{
    int maxIter;
    double mu;  //2-norm
    double l1Sparsity;
    double eta; //initial step-size
};

void AcceleratedProjGrad(const SparseMatrix& A,
    const Eigen::VectorXd& label, AcceleratedProjGradAlgoPara& para,
    vector<double>& tDiff, vector<double>& curLoss){
    SparseMatrix AT = A.transpose();
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
	Eigen::VectorXd x_s = Eigen::VectorXd::Zero(D);
	Eigen::VectorXd y_s(x_s);
	Eigen::VectorXd v_s(x_s);
	double theta=0;
	double L = 1.0/para.eta;
	double m = para.mu/N;
	// theta^2 * L = (1-theta) gamma + m * theta = gamma + (m-gamma) * theta
	// L theta^2 + (gamma-m) theta - gamma = 0
	// theta = (m-gamma)+sqrt((gamma-m)^2+4L gamma) /(2L)
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
		double gamma = theta*theta*L;
		double theta = (m-gamma + sqrt((gamma-m)*(gamma-m)+4*L*gamma))/2/L;
        ////////////////////// Primal ///////////////////////
		y_s = x_s + (theta*gamma)/(gamma+m*theta) * (v_s-x_s);
		Eigen::VectorXd grad = primal_grad_smooth_hinge_loss_reg(
				            A, AT, label, y_s, para.mu / N);
		Eigen::VectorXd x_sp = y_s - para.eta * grad;
        tStart = clock();
		x_sp = l1_proj(x_sp, para.l1Sparsity);
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        v_s = (1-1/theta) * x_s + (1/theta) * x_sp;
		x_s = Eigen::VectorXd(x_sp);
		/*x_s = l1_proj(x_s, para.l1Sparsity);
		y_s = Eigen::VectorXd(y_sp);
		l_sp = (1+sqrt(1+4*l_s*l_s))/2;
		g_s = (1-l_s)/l_sp;
		l_s = l_sp;*/
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_s, para.mu / N);
        cout << curIter<< "-th iter loss: " << loss << endl;
        // cout << "                                        ";
        // cout << x_s.lpNorm<1>();
        // cout << "                                        ";
        // cout << (x_s.array() != 0).count() << endl;
        curLoss[curIter] = loss;
    }
    cout << "prediction_accuracy: " << prediction_accuracy(A, label, x_s) << endl;
}






