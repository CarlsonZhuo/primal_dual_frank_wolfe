#include "DataLoader.h"
#include "util.h"
#include "FW.cpp"
#include "APG.cpp"
#include "SVRF.cpp"
#include "STORC.cpp"
#include "SVRG.cpp"
#include "SCGS.cpp"
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

using namespace std;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;

int main(int argc, char* argv[]){
    if (argc<2){
        cout << "Usage: "<<argv[0]<<" <method> "<<endl;
        exit(0);
    }
    int D = 47236;
    int N = 20245;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    char* data_file = "../data/rcv1_train.binary";
    DataLoader::load(data_file, A, label);

    // Eigen::VectorXd x = Eigen::VectorXd::Random(D);
    // cout << primal_grad_smooth_hinge_loss_reg(A, y, x, 0.00005) << endl;

    // Eigen::VectorXd x2(4);
    // x2 << -1,2,3,-4;
    // cout << l0_proj(x2, 2) << endl;;
    // cout << l1_proj(x2, 2) << endl;;
    if (strcmp(argv[1], "FW") == 0){
        FrankWolfeAlgoPara para;
        para.maxIter = 2000;
        para.mu = 10;
        para.l1Sparsity = 3000;
        para.eta = 0.5;

        vector<double> tDiff(para.maxIter);
        vector<double> curLoss(para.maxIter);
        FrankWolfe(A, label, para, tDiff, curLoss);
    }else if (strcmp(argv[1], "APG") == 0){
        AcceleratedProjGradAlgoPara para_apg;
        para_apg.maxIter = 500;
        para_apg.mu = 10;
        para_apg.l1Sparsity = 3000;
        para_apg.eta = 0.5;

        vector<double> tDiff(para_apg.maxIter);
        vector<double> curLoss(para_apg.maxIter);
        AcceleratedProjGrad(A, label, para_apg, tDiff, curLoss);
    }else if (strcmp(argv[1], "SVRF")==0){
		StochasticVRFAlgoPara para_svrf;
		para_svrf.maxIter = 50;
		para_svrf.mu = 10;
		para_svrf.l1Sparsity = 3000;
		para_svrf.eta = 0.5;

		RowMatrix rowA = A;
        vector<double> tDiff(para_svrf.maxIter);
		vector<double> curLoss(para_svrf.maxIter);
		StochasticVRF(rowA, label, para_svrf, tDiff, curLoss);
	}else if (strcmp(argv[1], "STORC")==0){
		StochasticTORCAlgoPara para;
		para.maxIter = 100;
		para.mu = 10;
		para.l1Sparsity = 3000;
		para.L = 10;

		RowMatrix rowA = A;
		vector<double> tDiff(para.maxIter);
        vector<double> curLoss(para.maxIter);
		StochasticTORC(rowA, label, para, tDiff, curLoss);
	}else if (strcmp(argv[1], "SVRG")==0){
		StochasticVRGAlgoPara para;
		para.maxIter = 100;
		para.mu = 10;
		para.l1Sparsity = 3000;
		RowMatrix rowA = A;
		vector<double> tDiff(para.maxIter);
		vector<double> curLoss(para.maxIter);
		StochasticVRG(rowA, label, para, tDiff, curLoss);
	}else if (strcmp(argv[1], "SCGS")==0){
		StochasticCGSAlgoPara para;
		para.maxIter = 1000;
		para.mu = 10;
		para.l1Sparsity = 300;
		para.L = 10;
		RowMatrix rowA = A;
		vector<double> tDiff(para.maxIter);
		vector<double> curLoss(para.maxIter);
		StochasticCGS(rowA, label, para, tDiff, curLoss);
	}else{
        cout << "Error method!!"<<endl;
        exit(0);
    }
}
