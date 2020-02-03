#include "Matrix.h"
#include "SolverUtil.h"
#include "Matrix.cpp"
#include "SolverUtil.cpp"
#include "SymmetricMatrix.h"
#include "SymmetricMatrix.cpp"
#include "SPDMatrix.h"
#include "SPDMatrix.cpp"
#include "BandedMat.h"
#include "BandedMat.cpp"
#include "CSRMatrix.h"
#include "CSRMatrix.cpp"
#include <algorithm>
#include <utility>
#include <iomanip>
using namespace util;
using std::cout;


int main() {


    //CSRMatrix Test/Example Usage
    //--------------------------------------------------------------------------------
    cout << "Using the CSRMatrix class to solve sparse postive definite systems: \n\n";
    double* vals = new double[21]{ 8,1,5,4,1,9,5,2,7,4,7,1,4,7,1,9,10,11,12,13,14};
    int* col_idx = new int[21]{ 0,1,2,5,0,1,0,2,5,3,4,5,0,2,4,5,6,7,8,9,10 };
    int* row_ps = new int[12]{ 0,4,6,9,10,12,16,17,18,19,20,21 };

    

    //11 by 11 sparse matrix with 21 non zero values
    CSRMatrix<double>my_csr{ 11,11,21,vals,row_ps, col_idx };
    //right hand side of equation
    std::vector<double>b_rhs{ 1,2,3,4,5,6,7,8,9,10,11 };
    //initial guess
    std::vector<double>x_init{ 0,0,0,0,0,0,0,0,0,0,0 };
    //solve method
    std::vector<double>soln = my_csr.solve(b_rhs, x_init);
    //print out solution
    for (auto p : soln)cout << p << " ";
    cout << "\n\n";

    ////Test to check if solution is correct
    ////b_init should be close to b_rhs
    //std::vector<double>b_init = my_csr.mat_vec_mult(soln);
    //cout << "Should be close to b_rhs: ";
    //for (auto p : b_init)cout<< p << " ";
    //cout << "\n\n";

    ////Check the error in the solution
    //std::vector<double>error(b_init.size());
    //std::transform(b_rhs.begin(), b_rhs.end(), b_init.begin(), error.begin(),
    //    [](double x, double y) {return std::fabs(x - y); });
    //cout << "Errors in solution: ";
    //for (auto p : error)cout<< p << " ";
    //cout << "\n\n\n\n";

    std::vector<double> ans = util::sp::CSR_cholesky(my_csr);

    for (auto p : ans)cout << p << " ";



    delete[] vals;
    delete[] col_idx;
    delete[] row_ps;

    //----------------------------------------------------------------------------------

   
    
    //Banded Matrix Test/Example Usage
    //----------------------------------------------------------------------------------
    cout << "Using the BandedMat class to solve banded matrix systems: \n\n";

    double* b_vals = new double[40]{ 4,3,9,6,7,2,4,8,3,2,7,8,22,34,11,5,4,
        7,9,5,6,4,9,10,12,14,19,20,13,7,3,24,11,2,5,6,7,8,4,10 };
    
    int lower_bandwidth = 1;
    int upper_band_width = 2;

    int rows = 11;
    int cols = 11;

    //11 by 11 banded matrix with upper bandwidth of 2 and lower bandwidth of 1
    BandedMat<double>my_bandedmat{ rows,cols,b_vals,lower_bandwidth, upper_band_width };
    //right hand side of equation
    std::vector<double>b_rhs2{ 1,2,3,4,5,6,7,8,9,10,11 };
    //solve method
    std::vector<double>soln2 = my_bandedmat.solve(b_rhs2);
    //print out solution
    for (auto p : soln2)cout << p << " ";
    cout << "\n\n";

    delete[] b_vals;

    //----------------------------------------------------------------------------------


    
    //Matrix(dense) Test/Example Usage
    //-----------------------------------------------------------------------------------
    cout << "Using the Matrix class to solve dense matrix systems: \n\n";
    double* m_vals = new double[121]{ 1,2,3,4,5,6,7,8,9,10,11,
                                    10,9,8,7,6,5,4,3,2,1,4,
                                    6,5,2,4,7,3,2,5,4,8,9,
                                    5,8,2,7,6,4,9,4,8,3,2,
                                    1,8,9,10,1,2,3,5,7,6,4,
                                    3,4,8,10,12,16,3,4,8,9,11,
                                    5,6,1,2,3,4,5,8,9,5,4,
                                    8,4,3,1,5,7,6,4,3,8,7,
                                    9,2,4,8,12,16,2,4,3,5,9,
                                    10,4,3,2,1,7,8,9,5,6,4,
                                    11,5,4,8,9,1,2,5,7,6,7 };
    
    //11 by 11 dense matrix
    Matrix<double>my_densemat{ 11,11,m_vals };
    //right hand side of equation
    std::vector<double>b_rhs3{ 1,2,3,4,5,6,7,8,9,10,11 };
    //solve method
    std::vector<double>soln3 = my_densemat.solve(b_rhs3);
    //print out solution
    for (auto p : soln3)cout << p << " ";
    cout << "\n\n";

    delete[] m_vals;
    //-----------------------------------------------------------------------------------


    //Symmetric Matrix(dense) Test/Example Usage
    //-----------------------------------------------------------------------------------
    cout << "Using the SymmetricMatrix class to solve dense symmetric(negative or positive) matrix systems: \n\n";
    long double* sym_vals = new long double[121]{ 1,2,3,4,5,6,7,8,9,10,11,
                                    2,9,8,7,6,5,4,3,2,1,4,
                                    3,8,2,4,7,3,2,5,4,8,9,
                                    4,7,4,7,6,4,9,4,8,3,2,
                                    5,6,7,6,1,2,3,5,7,6,4,
                                    6,5,3,4,2,16,3,4,8,9,11,
                                    7,4,2,9,3,3,5,8,9,5,4,
                                    8,3,5,4,5,4,8,4,4,9,5,
                                    9,2,4,8,7,8,9,4,3,5,7,
                                    10,1,8,3,6,9,5,9,5,6,6,
                                    11,4,9,2,4,11,4,5,7,6,7 };

    //11 by 11 dense symmetic matrix
    SymmetricMatrix my_symmat{ 11,11,sym_vals };

    //right hand side of equation
    std::vector<long double>b_rhs4{5,2,3,4,5,6,7,8,9,10,11 };
    //solve method
    std::vector<long double>soln4 = my_symmat.solve(b_rhs4);
    //print out solution
    cout << std::fixed;
    for (auto p : soln4)cout <<std::setprecision(20)<< p << " ";
    cout << "\n\n";

    delete[] sym_vals;

    //-----------------------------------------------------------------------------------


    
    //Symmetric Positive Definite(dense) Test/Example Usage
    //-----------------------------------------------------------------------------------
    cout << "Using the SPDMatrix class to solve dense positive symmetric matrix systems \n\n";
    long double* spd_vals = new long double[121]{ 20,2,3,4,5,6,7,8,9,10,11,
                                    2,19,8,7,6,5,4,3,2,1,4,
                                    3,8,12,4,7,3,2,5,4,8,9,
                                    4,7,4,17,6,4,9,4,8,3,2,
                                    5,6,7,6,21,2,3,5,7,6,4,
                                    6,5,3,4,2,16,3,4,8,9,11,
                                    7,4,2,9,3,3,15,8,9,5,4,
                                    8,3,5,4,5,4,8,14,4,9,5,
                                    9,2,4,8,7,8,9,4,33,5,7,
                                    10,1,8,3,6,9,5,9,5,29,6,
                                    11,4,9,2,4,11,4,5,7,6,27 };

    SPDMatrix my_spdmat{ 11,11,spd_vals };
    
    //right hand side of equation
    std::vector<long double>b_rhs5{ 5,2,3,4,5,6,7,8,9,10,11 };
    //solve method
    std::vector<long double>soln5 = my_spdmat.solve(b_rhs5);
    //print out solution
    cout << std::fixed;
    for (auto p : soln5)cout << std::setprecision(20) << p << " ";
    cout << "\n\n";
    delete[] spd_vals;
    //-----------------------------------------------------------------------------------

    cout << "NOTE: These solutions have been checked against scipy solvers and are seen to be accurate\n";
    
}