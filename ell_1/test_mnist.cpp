#include "DataLoader.h"
#include "util.h"
#include "pdFW.cpp"
#include "FW.cpp"
#include "APG.cpp"
#include "STORC.cpp"
#include "SVRG.cpp"
#include "SCGS.cpp"
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

using namespace std;

int main(int argc, char* argv[]){
    if (argc<2){
        cout << "Usage: "<<argv[0]<<" <method> "<<endl;
        exit(0);
    }
    int D = 894499;
    int N = 11872;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    char* mnist = "../data/mnist.09.bin";
    DataLoader::load(mnist, A, label);

    primalDualFrankWolfeAlgoPara para_pdFW;
    FrankWolfeAlgoPara para_FW;
    AcceleratedProjGradAlgoPara para_apg;
    StochasticTORCAlgoPara para_storc;
    StochasticVRGAlgoPara para_svrg;
    StochasticCGSAlgoPara para_scgs;

    // Common parameters
    double maxIter = 500;
    double mu = 10;
    double l1Sparsity = 300;
    para_pdFW.mu = mu;
    para_pdFW.l1Sparsity = l1Sparsity;
    para_FW.mu = mu;
    para_FW.l1Sparsity = l1Sparsity;
    para_apg.mu = mu;
    para_apg.l1Sparsity = l1Sparsity;
    
    para_storc.mu = mu;
    para_storc.l1Sparsity = l1Sparsity;
    para_svrg.mu = mu;
    para_svrg.l1Sparsity = l1Sparsity;
    para_scgs.mu = mu;
    para_scgs.l1Sparsity = l1Sparsity;
    char* csvFile;

    if (strcmp(argv[1], "pdFW") == 0){
        // PDFW
        para_pdFW.maxIter = 150;
        para_pdFW.l0Sparsity = 9000;
        para_pdFW.dualSparsity = 6000;
        para_pdFW.delta = 20;
        para_pdFW.eta = 0.5;
        para_pdFW.L = 2;
        //
        vector<double> pdFWTime(para_pdFW.maxIter);
        vector<double> pdFWLoss(para_pdFW.maxIter);
        primalDualFW(A, label, para_pdFW, pdFWTime, pdFWLoss);
        cout << "PDFW total time: " << sumVec(pdFWTime) << endl;
        cout << "PDFW final loss: " << pdFWLoss.back() << endl;
        csvFile = "./result/mnist09/pdFW";
        DataLoader::writeAsCSV(pdFWLoss, pdFWTime, csvFile);
    } else if (strcmp(argv[1], "FW") == 0){
        // FW
        para_FW.maxIter = maxIter * 5;
        para_FW.eta = 0.3;
        vector<double> FWTime(para_FW.maxIter);
        vector<double> FWLoss(para_FW.maxIter);
        FrankWolfe(A, label, para_FW, FWTime, FWLoss);
        csvFile = "./result/mnist09/FW";
        DataLoader::writeAsCSV(FWLoss, FWTime, csvFile);
    } else if (strcmp(argv[1], "APG") == 0){
        // APG
        para_apg.maxIter = 200;
        para_apg.eta = 80;
        //
        vector<double> APGTime(para_apg.maxIter);
        vector<double> APGLoss(para_apg.maxIter);
        AcceleratedProjGrad(A, label, para_apg, APGTime, APGLoss);
        csvFile = "./result/mnist09/APG";
        DataLoader::writeAsCSV(APGLoss, APGTime, csvFile);
    } else if (strcmp(argv[1], "STORC") == 0){
        // STORC
        para_storc.maxIter = 5;
        para_storc.L = 1;
        RowMatrix rowA = A;
        vector<double> STORCTime(para_storc.maxIter);
        vector<double> STORCLoss(para_storc.maxIter);
        StochasticTORC(rowA, label, para_storc, STORCTime, STORCLoss);
        csvFile = "./result/mnist09/STORC";
        DataLoader::writeAsCSV(STORCLoss, STORCTime, csvFile);
    } else if (strcmp(argv[1], "SVRG") == 0){
        // SVRG
        para_svrg.maxIter = 5;
        para_svrg.eta = 40;
        RowMatrix rowA = A;
        vector<double> SVRGTime(para_svrg.maxIter);
        vector<double> SVRGLoss(para_svrg.maxIter);
        StochasticVRG(rowA, label, para_svrg, SVRGTime, SVRGLoss);
        csvFile = "./result/mnist09/SVRG";
        DataLoader::writeAsCSV(SVRGLoss, SVRGTime, csvFile);
    } else if (strcmp(argv[1], "SCGS") == 0){
        // SVRG
        para_scgs.maxIter = 450;
        para_scgs.L = 5;
        RowMatrix rowA = A;
        vector<double> SCGSTime(para_scgs.maxIter);
        vector<double> SCGSLoss(para_scgs.maxIter);
        StochasticCGS(rowA, label, para_scgs, SCGSTime, SCGSLoss);
        csvFile = "./result/mnist09/SCGS";
        DataLoader::writeAsCSV(SCGSLoss, SCGSTime, csvFile);
    } else{
        cout << "Error method!!"<<endl;
        exit(0);
    }
}
