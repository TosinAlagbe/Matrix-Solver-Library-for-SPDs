#include <iostream>
#include "Matrix.h"
#include "SolverUtil.h"
#include <iomanip>
using std::cout;

// Constructor - using an initialisation list here
template <class T>
Matrix<T>::Matrix(int rows, int cols, bool preallocate)
    : rows(rows), cols(cols), size_of_values(rows* cols),
    preallocated(preallocate)
{
    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[size_of_values];
    }
}

// Constructor - now just setting the value of our T pointer
template <class T>
Matrix<T>::Matrix(int rows, int cols, T* values_ptr) :
    rows(rows), cols(cols), size_of_values(rows* cols),
    values(values_ptr) {}

//Copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix& mat)
    :rows{ mat.rows }, cols{ mat.cols },
    preallocated{ true }, values{ new T[mat.rows * mat.cols] }
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            values[i * cols + j] = mat.values[i * cols + j];
        }
    }
}

//copy assignment
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& mat) {

    //if (this->values)delete[] this->values;
    T* vals = new T[mat.rows * mat.cols];
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            vals[i * mat.cols + j] = mat.values[i * mat.cols + j];
        }
    }
    delete[] values;
    values = vals;
    rows = mat.rows;
    cols = mat.cols;
    return *this;
}

//matrix multiply
template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T>& mat) {

    Matrix<T> res{ rows,mat.cols };
    res.init_def(0);
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < mat.cols; ++k) {
            T sum = 0;
            for (int j = 0; j < cols; ++j) {
                sum += this->operator()(i, j) * mat(j, i);
            }
            res(i, k) = sum;
        }
    }
    return res;
}

// destructor
template <class T>
Matrix<T>::~Matrix()
{
    // Delete the values array
    if (this->preallocated) {
        delete[] this->values;
    }
}


// Just print out the values in our values array
template <class T>
void Matrix<T>::printValues()
{
    std::cout << "Printing values" << std::endl;
    for (int i = 0; i < this->size_of_values; i++)
    {
        std::cout << this->values[i] << " ";
    }
    std::cout << std::endl;
}

// Explicitly print out the values in values array
//as if they are a matrix
template <class T>
void Matrix<T>::printMatrix()
{
    std::cout << std::endl;
    for (int j = 0; j < this->rows; j++)
    {
        for (int i = 0; i < this->cols; i++)
        {
            // We have explicitly used a row-major ordering here 
            std::cout << std::fixed;
            std::cout << std::setprecision(3) << '\t' << this->values[i + j * this->cols] << " ";

        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::apply_all(const T& elem) {

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            values[i * cols + j] = elem;
        }
    }
}

template<typename T> 
void Matrix<T>::init_def(const T& elem) {
    if (!preallocated) {
        std::cout << "cannot call init on unowned array";
        return;
    }
    apply_all(elem);
}

template<typename T>
std::vector<T>  Matrix<T>::slice_col(int col_number) {

    std::vector<T>col(this->rows);
    for (int i = 0; i < this->rows; ++i) {
        col[i] = this->operator()(i, col_number);
    }
    return col;
}

template<typename T>
std::vector<T> Matrix<T>::slice_row(int row_number) {

    std::vector<T>col(this->cols);
    for (int i = 0; i < this->cols; ++i) {
        col[i] = this->operator()(row_number, i);
    }
    return col;
}


//vector becomes column of matrix at position col_num
template<typename T>
void Matrix<T>::set_col(int col_num, std::vector<T>& vals) {
    for (int i = 0; i < vals.size(); ++i)
        this->operator()(i, col_num) = vals[i];

}

template<class T>
Matrix<T> Matrix<T>::solve(Matrix<T>& b){

    //Matrix to store solution
    Matrix<T>x_matrix{ b.rows,b.cols };

    for (int k = 0; k < b.cols; ++k) {
        //split matrix b
        std::vector<T>b_slice = b.slice_col(k);

        //solve Ax = b for each b_slice
        bool decomposed = true;
        std::vector<T>x_vec = this->solve(b_slice, decomposed);
        x_matrix.set_col(k, x_vec);
    }
    return x_matrix;
  
}

template<class T>
std::vector<T> Matrix<T>::solve(std::vector<T>& b, bool decomposed){
    
    std::vector<T>soln = util::dense::GMRES(*this, b);
    return soln;
}

// Do matrix matrix multiplication
// output = this * mat_right
template <class T>
void Matrix<T>::matMatMult(Matrix& mat_right, Matrix& output)
{

    // Check our dimensions match
    if (this->cols != mat_right.rows)
    {
        std::cerr << "Input dimensions for matrices don't match" << std::endl;
        return;
    }

    // Check if our output matrix has had space allocated to it
    if (output.values != nullptr)
    {
        // Check our dimensions match
        if (this->rows != output.rows || this->cols != output.cols)
        {
            std::cerr << "Input dimensions for matrices don't match" << std::endl;
            return;
        }
    }
    // The output hasn't been preallocated, so we are going to do that
    else
    {
        output.values = new T[this->rows * mat_right.cols];
    }

    // Set values to zero before hand
    for (int i = 0; i < output.size_of_values; i++)
    {
        output.values[i] = 0;
    }

    // Now we can do our matrix-matrix multiplication
    // CHANGE THIS FOR LOOP ORDERING AROUND
    // AND CHECK THE TIME SPENT
    // Does the ordering matter for performance. Why??
    for (int i = 0; i < this->rows; i++)
    {
        for (int k = 0; k < this->cols; k++)
        {
            for (int j = 0; j < mat_right.cols; j++)
            {
                output.values[i * output.cols + j] += this->values[i * this->cols + k] * mat_right.values[k * mat_right.cols + j];
            }
        }
    }
}

template<class T>
Matrix<T> Matrix<T>::operator+(Matrix<T>& mat)
{
    Matrix<T> res{ mat.rows, mat.cols };
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res(i, j) = this->operator()(i, j) + mat(i, j);
        }
    }
    return res;
}

template<class T>
Matrix<T> Matrix<T>::operator+(Matrix<T> mat)
{
    Matrix<T> res{ mat.rows, mat.cols };
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res(i, j) = this->operator()(i, j) + mat(i, j);
        }
    }
    return res;
}

template<class T>
Matrix<T> Matrix<T>::operator-(Matrix<T>& mat)
{
    Matrix<T> res{ mat.rows, mat.cols };
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            res(i, j) = this->operator()(i, j) - mat(i, j);
        }
    }
    return res;
}

template<typename T>
Matrix<T> diag(int rows, int cols, const T& val) {

    Matrix<T> temp{ rows,cols };

    //initialise all to zero
    temp.init_def(0);

    for (int i = 0; i < rows; ++i) {
        temp(i, i) = val;
    }
    return temp;
}

template<typename T>
Matrix1d<T> matvecmul(const Matrix<T>& mat, const Matrix1d<T>& vec) {

    Matrix1d<T> res{ vec.rows };
    res.init_def(0);
    for (int i = 0; i < mat.rows; ++i) {
        T sum = 0;
        for (int j = 0; j < mat.cols; ++j) {
            sum += mat(i, j) * vec(j);
        }
        res(i) = sum;
    }
    return res;
}

template<typename T>
void fill_vec(Matrix1d<T>& mat, const T& val) {

    for (int i = 0; i < mat.rows; ++i) {
        mat(i) = val;
    }
}


