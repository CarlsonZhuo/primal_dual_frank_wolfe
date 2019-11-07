
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <cstring>
#include <iomanip>
#include "util.h"
#include <Eigen/Sparse>

using namespace std;


typedef Eigen::Triplet<double> Triplet;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SparseMatrix;
const int MAX_LENGTH = 1000000000;

class DataLoader{
    public:
    static void load(char* fileName,
                     SparseMatrix& X, 
                     Eigen::VectorXd& labels){
        ifstream fin(fileName);
        if( fin.fail() ){
            cerr << "cannot find file." << endl;
            exit(0);
        }
        // Prepare read file
        char* line = new char[MAX_LENGTH];
        vector<string> tokens;
        // Start reading
        int rowIdx = 0;
        vector<Triplet> XContent;
        while( !fin.eof() ){
            fin.getline(line, MAX_LENGTH);
            string str = string(line);
            tokens = split(str, " ");
            if( str.length() < 2 ){
                cout << "skip line " << rowIdx <<  endl;
                cout << str << endl;
                continue;
            }
            //yi
            double label = atof(tokens[0].c_str());
            if (label != 1 && label != -1){
                cout << "INPUT ERROR!" << endl;
                cout << str << endl;
                cout << "tokens[0] " << tokens[0].c_str() << endl;
                exit(0);
            }
            labels(rowIdx) = label;
            //xi
            for(int i = 1; i < tokens.size(); i++){
                vector<string> pv = split(tokens[i], ":");
                int colIdx = atoi(pv[0].c_str()) - 1;
                double val = atof(pv[1].c_str());
                XContent.push_back( Triplet(rowIdx, colIdx, val) );
            }
            rowIdx += 1;
        }
        X.setFromTriplets(XContent.begin(), XContent.end());
        delete[] line;
    }

    static vector<string> split(string str, string pattern){
        vector<string> str_split;
        size_t i = 0;
        size_t index = 0;
        while( index != string::npos ){
            index = str.find(pattern, i);
            str_split.push_back(str.substr(i, index-i));
            i = index + 1;
        }
        if( str_split.back() == "" )
            str_split.pop_back();
        return str_split;
    }

    static void genData(int d, int n, double l1Spar, double l0Spar, 
                        Eigen::VectorXd& w, Eigen::VectorXd& y, SparseMatrix& A){

        srand(2048);
        // srand((unsigned int) time(0));
        int rows = A.rows();
        int cols = A.cols();
        w.setRandom();
        w = l0_proj(w, l0Spar);
        w = l1_proj(w, l1Spar);
        //
        sparseMatGen(A);
        y = A * w;
        for(int i = 0; i < y.size(); i ++){
            if (y[i] >= 0)
                y[i] = 1;
            else
                y[i] = -1;
        }

    }

    static void writeAsCSV(vector<double>& loss, vector<double>& tDiff, char* fileName){
        ofstream csvFile;
        csvFile.open(fileName);
        int n = loss.size();
        for (int i = 0; i < n; i ++){
            csvFile << std::setprecision(15) << loss[i] << ", ";
        }
        csvFile << endl;
        for (int i = 0; i < n; i ++){
            csvFile << tDiff[i] << ", ";
        }
        csvFile.close();
    }

};
