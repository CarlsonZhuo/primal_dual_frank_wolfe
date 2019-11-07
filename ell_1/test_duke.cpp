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
    int D = 7129;
    int N = 44;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    char* dukeFile = "../data/duke";
    DataLoader::load(dukeFile, A, label);

    // Row Normalize
    // A = A.diagonal().asDiagonal().inverse() * A;
    Eigen::VectorXd onesVec(D);
    for(int i = 0; i < D; i ++)
        onesVec[i] = 1;
    Eigen::VectorXd rowSum = A * onesVec;
    A = rowSum.asDiagonal().inverse() * A;

    primalDualFrankWolfeAlgoPara para_pdFW;
    FrankWolfeAlgoPara para_FW;
    AcceleratedProjGradAlgoPara para_apg;
    StochasticTORCAlgoPara para_storc;
    StochasticVRGAlgoPara para_svrg;
    StochasticCGSAlgoPara para_scgs;

    // Common parameters
    double mu = 10;
    double l1Sparsity = 0.5;
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
        para_pdFW.maxIter = 100;
        para_pdFW.l0Sparsity = 500;
        para_pdFW.dualSparsity = 44;
        para_pdFW.delta = 20;
        para_pdFW.eta = 0.5;
        para_pdFW.L = 1;
        //
        vector<double> pdFWTime(para_pdFW.maxIter);
        vector<double> pdFWLoss(para_pdFW.maxIter);
        primalDualFW(A, label, para_pdFW, pdFWTime, pdFWLoss);
        cout << "PDFW total time: " << sumVec(pdFWTime) << endl;
        cout << "PDFW final loss: " << pdFWLoss.back() << endl;
        csvFile = "./result/duke/pdFW";
        DataLoader::writeAsCSV(pdFWLoss, pdFWTime, csvFile);
    } else if (strcmp(argv[1], "FW") == 0){
        // FW
        para_FW.maxIter = 500;
        para_FW.eta = 0.5;
        vector<double> FWTime(para_FW.maxIter);
        vector<double> FWLoss(para_FW.maxIter);
        FrankWolfe(A, label, para_FW, FWTime, FWLoss);
        csvFile = "./result/duke/FW";
        DataLoader::writeAsCSV(FWLoss, FWTime, csvFile);
    } else if (strcmp(argv[1], "APG") == 0){
        // APG
        para_apg.maxIter = 100;
        para_apg.eta = 1.5;
        //
        vector<double> APGTime(para_apg.maxIter);
        vector<double> APGLoss(para_apg.maxIter);
        AcceleratedProjGrad(A, label, para_apg, APGTime, APGLoss);
        csvFile = "./result/duke/APG";
        DataLoader::writeAsCSV(APGLoss, APGTime, csvFile);
    } else if (strcmp(argv[1], "STORC") == 0){
        // STORC
        para_storc.maxIter = 10;
        para_storc.smallData = 1;
        para_storc.L = 0.1;
        RowMatrix rowA = A;
        vector<double> STORCTime(para_storc.maxIter);
        vector<double> STORCLoss(para_storc.maxIter);
        StochasticTORC(rowA, label, para_storc, STORCTime, STORCLoss);
        csvFile = "./result/duke/STORC";
        DataLoader::writeAsCSV(STORCLoss, STORCTime, csvFile);
    } else if (strcmp(argv[1], "SVRG") == 0){
        // SVRG
        para_svrg.maxIter = 5;
        para_svrg.smallData = 1;
        para_svrg.eta = 5;
        RowMatrix rowA = A;
        vector<double> SVRGTime(para_svrg.maxIter);
        vector<double> SVRGLoss(para_svrg.maxIter);
        StochasticVRG(rowA, label, para_svrg, SVRGTime, SVRGLoss);
        csvFile = "./result/duke/SVRG";
        DataLoader::writeAsCSV(SVRGLoss, SVRGTime, csvFile);
    }  else if (strcmp(argv[1], "SCGS") == 0){
        // SVRG
        para_scgs.maxIter = 50;
        para_scgs.L = 0.5;
        RowMatrix rowA = A;
        vector<double> SCGSTime(para_scgs.maxIter);
        vector<double> SCGSLoss(para_scgs.maxIter);
        StochasticCGS(rowA, label, para_scgs, SCGSTime, SCGSLoss);
        csvFile = "./result/duke/SCGS";
        DataLoader::writeAsCSV(SCGSLoss, SCGSTime, csvFile);
    } else{
        cout << "Error method!!"<<endl;
        exit(0);
    }
}
