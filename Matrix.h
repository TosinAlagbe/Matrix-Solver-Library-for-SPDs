#ifndef MATRIX_H
#define MATRIX_H

#include<iostream>
#include<vector>
#include <memory>
#include <algorithm>
#include <utility>
#include <functional>
#include <string>



template<typename T>
class DecompMatrix;

//Dense Matrix Class
//Used to solve any class of Matrix
//Used this only if you cannot ascertain the class of your Matrix
//Might be inefficient for some classes of Matrices
template <class T>
class Matrix
{
public:

    // constructor where we want to preallocate ourselves
    //rows: row size of dense matrix
    //cols: col size of dense matrix
    //preallocate: boolean to denote allocation 
    Matrix(int rows, int cols, bool preallocate = true);
    
    // constructor where we already have allocated memory outside
    //rows: row size of dense matrix
    //cols: col size of dense matrix
    //values_ptr: one-d array of dense matrix values traversed row-wise 
    Matrix(int rows, int cols, T* values_ptr);

    //default constructor
    Matrix() :rows{ 0 }, cols{ 0 }, values{ nullptr } {};
    //copy constructor
    Matrix(const Matrix& mat);

    //copy assignment
    Matrix& operator=(const Matrix&);

    // destructor
    virtual ~Matrix();

    // Print out the values in our matrix
    void printValues();
    virtual void printMatrix();

    //Fill matrix with value: elem
    void apply_all(const T& elem);


    //initialise with constant value
    virtual void init_def(const T& elem);
  
    //get column of col_number from matrix
    std::vector<T> slice_col(int col_number);

    //get row of row_number from matrix
    std::vector<T> slice_row(int row_number);

    //vector becomes column of matrix at position col_num
    void set_col(int col_num, std::vector<T>& vals);

    //solver: uses the GMRES algorithm as default
    //b: Matrix of multiple right hand sides
    //decomposed: boolean to know if matrix is already decomposed
    virtual std::vector<T> solve(std::vector<T>& b, bool decomposed = false);
    //solver: uses the GMRES algorithm as default
    //b: right hand side vector
    virtual Matrix solve(Matrix& b);

    //create diagonal matrix
    friend Matrix diag(int rows, int cols, const T& val);

    // Matrix-Matrix multiplication
    virtual void matMatMult(Matrix& mat_right, Matrix& output);


    //2d indexing
    virtual T& operator()(int i, int j) { return this->values[i * cols + j]; }

    //overload operators for matrix addition
    Matrix operator+(Matrix&);
    Matrix operator+(Matrix);

    //overload operators for matrix multiply
    Matrix operator*(Matrix&);

    //overload - operator for matrix subtraction
    Matrix operator-(Matrix&);

    //Explicitly using the C++11 nullptr here
    T* values = nullptr;
    int rows = -1;
    int cols = -1;

    // Private variables - there is no need for other classes 
    // to know about these variables
private:
    int size_of_values = -1;


protected:
    bool preallocated = false;
    
};


//declarations
template<typename T>
class Matrix1d;

template<typename T>
Matrix<T> diag(int rows, int cols, const T& val);

template<typename T>
Matrix1d<T> matvecmul(const Matrix<T>& mat, const Matrix1d<T>& vec);

template<typename T>
void fill_vec(Matrix1d<T>& mat, const T& val);


//1d Matrix (Could also use vector instead)
template< typename T>
class Matrix1d : public Matrix<T> {

public:
    Matrix1d() {}
    Matrix1d(int rows)
    {
        this->rows = rows;
        //1d matrix
        this->cols = 1;
        if (this->values)delete[] this->values;
        this->values = new T[rows];
    }

    //copy assignment
    Matrix1d& operator=(const Matrix1d& mat) {
        T* vals = new T[mat.rows];
        for (int i = 0; i < mat.rows; ++i) {
            vals[i] = mat.values[i];
        }
        delete[] this->values;
        this->values = vals;
        this->rows = mat.rows;
        this->cols = mat.cols;
        return *this;
    }

    T& operator()(int i, int j = 0) override
    {
        return this->values[i];
    }

    void init_def(const T& elem) {
        for (int i = 0; i < this->rows; ++i)
            this->values[i] = elem;
    }

    void printMatrix() override {
        for (int i = 0; i < this->rows; ++i) {
            std::cout << this->values[i] << " ";
        }
        std::cout << "\n";
    }
};

//Container to hold decomposed matrices
template<typename T>
class DecompMatrix {

public:

    //default constructor
    DecompMatrix() :lower(), diag(), upper() {}

    //constructor with lower and upper triangular matrix
    DecompMatrix(Matrix<T>& l, Matrix<T>& u)
        :lower{ l }, upper{ u }{}

    //constructor with lower traingular and diagonal matrix
    DecompMatrix(Matrix<T>& l, Matrix1d<T>& d)
        :lower{ l }, diag{ d }{}

    //copy assignment
    DecompMatrix& operator=(const DecompMatrix& decomp) {
        this->lower = decomp.lower;
        this->diag = decomp.diag;
        this->upper = decomp.upper;
        return *this;
    }

    //constructor with only lower triangular matrix
    DecompMatrix(Matrix<T>& l) :lower{ l } {}

    //lower triangular matrix
    Matrix<T> lower;

    //diagonal matrix as vector
    Matrix1d<T>diag;

    //upper triangular matrix
    Matrix<T> upper;
};


#endif // !MATRIX_H