#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/Dense>

using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> RowMatrix;
typedef Eigen::Triplet<double> T;


double sumVec(const vector<double>& vec){
    double result = 0;
    int size = vec.size();
    for(int i = 0; i < size; i ++){
        result += vec[i];
    }
    return result;
}


Eigen::VectorXd simplex_proj(const Eigen::VectorXd& x, double s){
    /*
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    */
    // Check if already within the simplex
    int n = x.rows();
    bool allPos = 1;
    for (int i = 0; i < n; i ++){
        if (x(i) < 0) { allPos = 0; break; }
    }
    if (x.sum() <= s && allPos)
        return x;
    // Copy and sort in decreasing order
    Eigen::VectorXd u(x);
    std::sort(u.data(), u.data() + n, std::greater<double>());
    // Compute cusum of sorted x
    Eigen::VectorXd cssx(n);
    cssx(0) = u(0);
    for (int i = 1; i < n; i ++)
        cssx(i) = cssx(i-1) + u(i);
    // get the number of > 0 components of the optimal solution
    int rho = -1;
    for (int i = 0; i < n; i ++){
        if (u(i) * (i + 1) > cssx(i) - s)
            rho = i;
    }
    // compute the Lagrange multiplier associated to the simplex constraint
    double theta = 0;
    theta = (cssx(rho) - s) / (rho + 1);
    // compute the simplex proj
    Eigen::VectorXd result(n);
    for (int i = 0; i < n; i ++){
        if (x(i) - theta <= 0)
            result(i) = 0;
        else
            result(i) = x(i) - theta;
    }
    return result;
}


Eigen::VectorXd l1_proj(const Eigen::VectorXd& x, double s){
    if (x.lpNorm<1>() <= s)
        return x;
    Eigen::VectorXd w = simplex_proj(x.cwiseAbs(), s);
    // w *= sign(x)
    for(int i = 0; i < x.rows(); i++){
        double sign_xi = (2 * (x(i) > 0) - 1);
        w(i) = w(i) * sign_xi;
    }
    return w;
}


Eigen::VectorXd l0_proj(const Eigen::VectorXd& x, int s){
    if ((x.array() != 0).count() <= s)
        return x;
    // Copy, and find the s-largest
    Eigen::VectorXd u(x.cwiseAbs());
    //
    std::nth_element(
        u.data(),
        u.data() + s - 1,
        u.data() + u.rows(),
        std::greater<double>());
    double sLargest = u(s - 1);
    //
    // Iterate over the result
    Eigen::VectorXd result(x.rows());
    result.setZero();
    int selectedCNT = 0;
    for(int i = 0; i < x.rows(); i ++){
        if (std::abs(x[i]) >= sLargest){
            result[i] = x[i];
            selectedCNT += 1;
        }
        if (selectedCNT == s)
            break;
    }
    return result;

}


double smooth_hinge_loss(const SparseMatrix& A, const Eigen::VectorXd& y,
                         const Eigen::VectorXd& x){
    int N = A.rows();
    Eigen::VectorXd w = A * x;
    Eigen::VectorXd wy = w.cwiseProduct(y);
    double loss = 0;
    for (int i = 0; i < N; i ++){
        if (wy(i) <= 0){
            loss += 0.5 - wy(i);
        }
        else if (wy(i) <= 1){
            loss += 0.5 * (1 - wy(i)) * (1 - wy(i));
        }
    }
    loss /= N;
    return loss;
}


double smooth_hinge_loss_reg(const SparseMatrix& A, const Eigen::VectorXd& y,
                             const Eigen::VectorXd& x, double mu){
    double loss = smooth_hinge_loss(A, y, x);
    double reg_loss = 0.5 * mu * x.squaredNorm();
    return loss + reg_loss;
}


double prediction_accuracy(const SparseMatrix& A, const Eigen::VectorXd& y,
                           const Eigen::VectorXd& x){
    int N = A.rows();
    double correct_cnt = 0;
    Eigen::VectorXd w = A * x;
    for (int i = 0; i < N; i ++){
        if (w(i) * y(i) > 0)
            correct_cnt += 1;
    }
    return correct_cnt / N;
}


Eigen::VectorXd primal_grad_smooth_hinge_loss_reg(
            const SparseMatrix& A,
            const SparseMatrix& AT,
            const Eigen::VectorXd& y,
            const Eigen::VectorXd& x, double mu){
    int N = A.rows();
    int D = A.cols();
    Eigen::VectorXd w = A * x;
    Eigen::VectorXd wy = w.cwiseProduct(y);
    Eigen::VectorXd gradient1(N);
    for (int i = 0; i < N; i ++){
        if (wy(i) <= 0){
            gradient1(i) = - y(i);
        }
        else if (wy(i) <= 1){
            gradient1(i) = (wy(i) - 1) * y(i);
        }
		else
			gradient1(i)=0;
    }
    Eigen::VectorXd gradient = (AT * gradient1) / N + mu * x;
    return gradient;
}


Eigen::VectorXd primal_grad_smooth_hinge_loss_reg(
		const RowMatrix& A, const Eigen::VectorXd& y,
		const Eigen::VectorXd& x, double mu,
		int k, vector<int>& indices){
	int D = A.cols();
	Eigen::VectorXd w(k);
	// compute w<==A[indices,:]*x
	for (int i=0;i<k;i++){
		w[i] = 0;
    	for (RowMatrix::InnerIterator it(A, indices[i]);it;++it){
			w[i] += it.value() * x[it.col()];
		}
	}
	Eigen::VectorXd wy(k);
	for (int i=0;i<k;i++){
    	wy[i] = w[i] * y[indices[i]];
	}
	Eigen::VectorXd gradient1(k);
	for (int i = 0; i < k; i ++){
	    if (wy[i] <= 0){
	        gradient1[i] = - y[indices[i]];
	    }
	    else if (wy(i) <= 1){
	        gradient1[i] = (wy(i) - 1) * y[indices[i]];
	    }
		else
			gradient1[i]=0;
	}
	Eigen::VectorXd gradient = Eigen::VectorXd::Zero(D);
	for (int i=0; i<k; i++){
		for (RowMatrix::InnerIterator it(A, indices[i]);it;++it){
			gradient[it.col()] += it.value() * gradient1(i);
		}
	}
	gradient = gradient/k + mu*x;
	return gradient;
}


double simplex_proj_find_theta(const vector<double>& x, double s){
    int n = x.size();
    bool allPos = 1;
    for (int i = 0; i < n; i ++){
        if (x[i] < 0) { allPos = 0; break; }
    }
    double sumX = sumVec(x);
    if (sumX <= s && allPos)
        return 0;
    // Copy and sort in decreasing order
    vector<double> u(x.begin(), x.end());
    std::sort(u.begin(), u.end(), std::greater<double>());
    // Compute cusum of sorted x
    vector<double> cssx(n);
    cssx[0] = u[0];
    for (int i = 1; i < n; i ++)
        cssx[i] = cssx[i-1] + u[i];
    // get the number of > 0 components of the optimal solution
    int rho = -1;
    for (int i = 0; i < n; i ++){
        if (u[i] * (i + 1) > cssx[i] - s)
            rho = i;
    }
    // compute the Lagrange multiplier associated to the simplex constraint
    double theta = 0;
    theta = (cssx[rho] - s) / (rho + 1);
    return theta;
}



Eigen::VectorXd l1_l0_proj(const Eigen::VectorXd& x, int s, double tau){
    if ((x.array() != 0).count() <= s)
        return l1_proj(x, tau);
    // Copy, and find the s-largest
    int n = x.rows();
    Eigen::VectorXd u(x.cwiseAbs());
    //
    std::nth_element(
        u.data(),
        u.data() + s - 1,
        u.data() + u.rows(),
        std::greater<double>());
    double sLargest = u(s - 1);
    //
    // Iterate over the result
    vector<double> compact_l0x(s);
    vector<double> compact_l0x_abs(s);
    vector<int> compact_l0x_idx(s);
    // Eigen::VectorXd l0x(x);
    Eigen::VectorXd l0x(n);
    l0x.setZero();
    int selectedCNT = 0;
    for(int i = 0; i < n; i ++){
        double val = x[i];
        if (abs(val) >= sLargest){
            l0x[i] = val;
            compact_l0x[selectedCNT] = val;
            compact_l0x_abs[selectedCNT] = abs(val);
            compact_l0x_idx[selectedCNT] = i;
            selectedCNT += 1;
        }
        if (selectedCNT == s)
            break;
    }
    // TODO
    // Input to l1_proj: l0x
    if (l0x.lpNorm<1>() <= tau)
        return l0x;
    // Simplex proj
    double theta = simplex_proj_find_theta(compact_l0x_abs, tau);
    Eigen::VectorXd result(n);
    result.setZero();
    for (int i = 0; i < s; i ++){
        int idx = compact_l0x_idx[i];
        if (compact_l0x_abs[i] - theta <= 0)
            result(idx) = 0;
        else
            result(idx) = compact_l0x_abs[i] - theta;
    }
    // l1 proj
    for(int i = 0; i < s; i++){
        int idx = compact_l0x_idx[i];
        double sign_idx = (2 * (l0x(idx) > 0) - 1);
        // double sign_idx = (2 * (compact_l0x_abs[i] > 0) - 1);
        result(idx) *= sign_idx;
    }
    return result;
    // return l1_proj(l0x, tau);
}



void normalMat(SparseMatrix& SM){
    int rows = SM.rows();
    int cols = SM.cols();
    Eigen::VectorXd onesVec(cols);
    for(int i = 0; i < cols; i ++)
        onesVec[i] = 1;
    Eigen::VectorXd rowNorm = SM.cwiseProduct(SM) * onesVec;
    rowNorm = rowNorm.cwiseSqrt();
    SM = rowNorm.asDiagonal().inverse() * SM;
}

void sparseMatGen(SparseMatrix& SM){
    int rows = SM.rows();
    int cols = SM.cols();
    std::vector<Eigen::Triplet<double> > tripletList;
    for(int i = 0; i < rows; i++){
        bool anySelected = 0;
        for(int j = 0; j < cols; j++)
        {
           auto v_ij = rand() % 100;
           if(v_ij < 20){
               tripletList.push_back(T(i, j, v_ij));
               anySelected = 1;
           }
        }
        if (! anySelected){
            // at least one non zero
            int j = rand() % cols;
            auto v_ij = rand() % 20;
            tripletList.push_back(T(i, j, v_ij));
        }
    }
    SM.setFromTriplets(tripletList.begin(), tripletList.end());   //create the matrix
    normalMat(SM);
}

#endif
