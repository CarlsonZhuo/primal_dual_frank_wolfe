#include "util.h"
#include <iostream>
#include <Eigen/Dense>
#include <iomanip>
#include <time.h>

using Eigen::MatrixXd;

struct primalDualFrankWolfeAlgoPara{
    int maxIter;
    double mu;
    int l0Sparsity;
    double l1Sparsity;
    double dualSparsity;
    double delta;  // dual learning rate
    double eta;
    double L;
};

Eigen::VectorXd grad_f_star(const Eigen::VectorXd& y, const Eigen::VectorXd& label){
    Eigen::VectorXd result = y + label;
    for (int i = 0; i < result.rows(); i ++){
        double tmp = y(i) * label(i);
        if ( tmp >= 0 or tmp <= -1 )
            result(i) = 0;
    }
    return result;
}

void primalDualFW(const SparseMatrix& A, const Eigen::VectorXd& label,
                  primalDualFrankWolfeAlgoPara& para,
                  vector<double>& tDiff, vector<double>& curLoss){
    int N = A.rows();  // Number of samples
    int D = A.cols();  // Number of dimensions
    // TODO: check if still row major!
    SparseMatrix AT = Eigen::SparseMatrix<double, Eigen::ColMajor>(A.transpose());
    // x: primal var; y: dual var
    Eigen::VectorXd x_i(D);
    x_i.setZero();
    Eigen::VectorXd y_i = Eigen::VectorXd::Random(N);
    // w: A * x; z: A.T * y
    Eigen::VectorXd w_i(N);
    Eigen::VectorXd z_i = AT * y_i ;
    Eigen::VectorXd delta_y(N);
    //
    clock_t tStart;
    for (int curIter = 0; curIter < para.maxIter; curIter ++){
        tStart = clock();
        ////////////////////// Primal ///////////////////////
        Eigen::VectorXd grad_x_L = z_i + para.mu * x_i;
        Eigen::VectorXd update_x = x_i - 1/(para.mu * para.L) * grad_x_L;
        // Eigen::VectorXd delta_x = l1_proj(l0_proj(update_x, para.l0Sparsity), para.l1Sparsity);
        Eigen::VectorXd delta_x = l1_l0_proj(update_x, para.l0Sparsity, para.l1Sparsity);
        x_i = (1 - para.eta) * x_i + para.eta * delta_x;
        // Update w
        w_i = (1 - para.eta) * w_i + para.eta * (A * delta_x.sparseView()).toDense();
        /////////////////////// Dual ///////////////////////
        Eigen::VectorXd coordin_selector = w_i - grad_f_star(y_i, label);
        coordin_selector = l0_proj(coordin_selector, para.dualSparsity);
        delta_y.setZero();
        for (int j = 0; j < N; j ++){
            if (coordin_selector(j) == 0) continue;
            double new_y_j_pos = (y_i[j] - para.delta + para.delta * w_i(j)) / (para.delta + 1);
            double new_y_j_neg = (y_i[j] + para.delta + para.delta * w_i(j)) / (para.delta + 1);
            double new_y_j;
            if (label(j) == -1)
                new_y_j = std::min(std::max(new_y_j_neg, 0.0), 1.0);
            else if (label(j) == 1)
                new_y_j = std::min(std::max(new_y_j_pos, -1.0), 0.0);
            else{
                cout << "label" << j << " is " << label(j) << endl;
                cout << "INPUT ERROR!" << endl;
                exit(0);
            }
            delta_y(j) = new_y_j - y_i(j);
            y_i(j) = new_y_j;
        }
        z_i = z_i + (AT * delta_y.sparseView()).toDense();
        ////////////////////// END ///////////////////////
        tDiff[curIter] = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        double loss = smooth_hinge_loss_reg(A, label, x_i, para.mu / N);
        curLoss[curIter] = loss;
        cout << loss << endl;
        // cout << "                                        ";
        // cout << x_i.lpNorm<1>();
        // cout << "                                        ";
        // cout << (x_i.array() != 0).count() << endl;
    }
    cout << "prediction_accuracy" << prediction_accuracy(A, label, x_i) << endl;
}

