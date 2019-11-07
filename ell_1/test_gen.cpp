#include "DataLoader.h"
#include "util.h"
#include "pdFW.cpp"
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

using namespace std;


int test1000(){
    int D = 1000;
    int N = 1000;
    double l1Sparsity = 5;
    double l0Sparsity = 5;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    Eigen::VectorXd w(D);
    DataLoader::genData(D, N, l1Sparsity, l0Sparsity, w, label, A);

    primalDualFrankWolfeAlgoPara para_pdFW;

    // Common parameters
    para_pdFW.mu = 1;
    para_pdFW.l1Sparsity = 10;
    para_pdFW.maxIter = 300;
    para_pdFW.l0Sparsity = 50;
    para_pdFW.dualSparsity = 300;
    para_pdFW.delta = 20;
    para_pdFW.eta = 0.25;
    para_pdFW.L = 3;
    //
    vector<double> pdFWTime(para_pdFW.maxIter);
    vector<double> pdFWLoss(para_pdFW.maxIter);
    primalDualFW(A, label, para_pdFW, pdFWTime, pdFWLoss);
    cout << "PDFW total time: " << sumVec(pdFWTime) << endl;
    cout << "PDFW final loss: " << pdFWLoss.back() << endl;
    char* csvFile = "./result/gen/pdFW1000";
    DataLoader::writeAsCSV(pdFWLoss, pdFWTime, csvFile);
}



int test3000(){
    int D = 3000;
    int N = 1000;
    double l1Sparsity = 5;
    double l0Sparsity = 5;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    Eigen::VectorXd w(D);
    DataLoader::genData(D, N, l1Sparsity, l0Sparsity, w, label, A);

    primalDualFrankWolfeAlgoPara para_pdFW;

    // Common parameters
    para_pdFW.mu = 1;
    para_pdFW.l1Sparsity = 10;
    para_pdFW.maxIter = 300;
    para_pdFW.l0Sparsity = 50;
    para_pdFW.dualSparsity = 700;
    para_pdFW.delta = 20;
    para_pdFW.eta = 0.25;
    para_pdFW.L = 3;
    //
    vector<double> pdFWTime(para_pdFW.maxIter);
    vector<double> pdFWLoss(para_pdFW.maxIter);
    primalDualFW(A, label, para_pdFW, pdFWTime, pdFWLoss);
    cout << "PDFW total time: " << sumVec(pdFWTime) << endl;
    cout << "PDFW final loss: " << pdFWLoss.back() << endl;
    char* csvFile = "./result/gen/pdFW3000";
    DataLoader::writeAsCSV(pdFWLoss, pdFWTime, csvFile);
}


int test9000(){
    int D = 9000;
    int N = 1000;
    double l1Sparsity = 5;
    double l0Sparsity = 5;
    SparseMatrix A(N, D);
    Eigen::VectorXd label(N);
    Eigen::VectorXd w(D);
    DataLoader::genData(D, N, l1Sparsity, l0Sparsity, w, label, A);

    primalDualFrankWolfeAlgoPara para_pdFW;

    // Common parameters
    para_pdFW.mu = 1;
    para_pdFW.l1Sparsity = 10;
    para_pdFW.maxIter = 300;
    para_pdFW.l0Sparsity = 50;
    para_pdFW.dualSparsity = 2000;
    para_pdFW.delta = 20;
    para_pdFW.eta = 0.25;
    para_pdFW.L = 3;
    //
    vector<double> pdFWTime(para_pdFW.maxIter);
    vector<double> pdFWLoss(para_pdFW.maxIter);
    primalDualFW(A, label, para_pdFW, pdFWTime, pdFWLoss);
    cout << "PDFW total time: " << sumVec(pdFWTime) << endl;
    cout << "PDFW final loss: " << pdFWLoss.back() << endl;
    char* csvFile = "./result/gen/pdFW9000";
    DataLoader::writeAsCSV(pdFWLoss, pdFWTime, csvFile);
}



int main(){
    test1000();
    test3000();
    test9000();
    return 1;
}
