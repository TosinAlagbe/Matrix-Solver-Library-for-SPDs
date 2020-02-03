#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include "Matrix.h"

//Compressed Sparse Row Matrix Class
//For efficient solving of sparse positive definite matrices
//Uses the efficient preconditioned conjugate gradient method
template <class T>
class CSRMatrix : public Matrix<T>
{
public:

    // constructor where we want to preallocate ourselves
    //rows: row size of dense form
    //cols: col size of dense form
    //nnzs: number of non zeros in sparse matrix
    //preallocate: boolean to denote allocation 
    CSRMatrix(int rows, int cols, int nnzs, bool preallocate);

    // constructor where we already have allocated memory outside
    //rows: row size of dense form
    //cols: col size of dense form
    //nnzs: number of non zeros in sparse matrix
    //values_ptr: one-d array of non-zero values
    //preallocate: boolean to denote allocation 
    CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index);
    
    //default constructor
    CSRMatrix();
    
    //copy constructor
    CSRMatrix(const CSRMatrix<T>& csr);
    //move constructor to return from function
    CSRMatrix(CSRMatrix<T>&& mat);

  
    // destructor
    ~CSRMatrix();

    // Print out the values in our matrix
    virtual void printMatrix() override;

    //solver: uses preconditioned conjugate gradient as default
    //uses Jacobi preconditioner
    //b: Matrix of multiple right hand sides
    //x_init: vector of initial guess of solution (use vector of zeros if not sure)
    //preconditioner: string specifiying type of preconditioner used
    //tol: tolerance value to control convergence
    //max_iter: maximum number of allowable iterations before failure
    virtual std::vector<T> solve(std::vector<T>& b, std::vector<T>& x_init,
        std::string preconditioner = "jacobi", double tol = 0.000001, int max_iter = 10000);
    
    //b: right hand side vector
    //x_init: vector of initial guess of solution (use vector of zeros if not sure)
    //preconditioner: string specifiying type of preconditioner used
    //tol: tolerance value to control convergence
    //max_iter: maximum number of allowable iterations before failure
    virtual Matrix<T> solve(Matrix<T>& b, std::vector<T>& x_init,
        std::string preconditioner = "jacobi", double tol = 0.00001, int max_iter = 10000);


    //get location of diagonal value on values array for each row
    std::vector<int> get_diags();
   

    // Perform some operations with our matrix
    void matMatMult(CSRMatrix<T>& mat_right, CSRMatrix<T>& output);
    // Perform some operations with our matrix
    void matVecMult(double* input, double* output);
    std::vector<T> mat_vec_mult(std::vector<T>x);

    //transpose of CSR
    CSRMatrix<T> transpose();

    // Explicitly using the C++11 nullptr here
    int* row_position = nullptr;
    int* col_index = nullptr;

    // How many non-zero entries we have in the matrix
    int nnzs = -1;

    // Private variables - there is no need for other classes 
    // to know about these variables
private:

};




















#endif // !CSR_MATRIX_H
