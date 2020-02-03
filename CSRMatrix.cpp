#include "CSRMatrix.h"
#include "SolverUtil.h"

// Constructor - using an initialisation list here
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, bool preallocate) : Matrix<T>(rows, cols, false), nnzs(nnzs)
{
    // If we don't pass false in the initialisation list base constructor, it would allocate values to be of size
    // rows * cols in our base matrix class
    // So then we need to set it to the real value we had passed in
    this->preallocated = preallocate;

    // If we want to handle memory ourselves
    if (this->preallocated)
    {
        // Must remember to delete this in the destructor
        this->values = new T[this->nnzs];
        this->row_position = new int[this->rows + 1];
        this->col_index = new int[this->nnzs];
    }
}

// Constructor - now just setting the value of our T pointer
template <class T>
CSRMatrix<T>::CSRMatrix(int rows, int cols, int nnzs, T* values_ptr, int* row_position, int* col_index) : Matrix<T>(rows, cols, values_ptr), nnzs(nnzs), row_position(row_position), col_index(col_index)
{}

//default constructor
template<class T>
CSRMatrix<T>::CSRMatrix()
    :row_position{ nullptr }, col_index{ nullptr }
{
    this->rows = 0;
    this->cols = 0;
    this->nnzs - 0;
    this->values = nullptr;
}


//Copy constructor
template<class T>
CSRMatrix<T>::CSRMatrix(const CSRMatrix & mat)
    : Matrix<T>(mat.rows, mat.cols, false)
{   
    this->rows = mat.rows;
    this->cols = mat.cols;
    this->nnzs = mat.nnzs;
    this->preallocated = mat.preallocated;

    if (this->preallocated) {
        this->values = new T[nnzs];
        this->row_position = new int[this->rows + 1];
        this->col_index = new int[this->nnzs];
    }
    for (int i = 0; i < this->nnzs; ++i) {
        this->values[i] = mat.values[i];
    }
    for (int i = 0; i < this->nnzs; ++i) {
        this->col_index[i] = mat.col_index[i];
    }
    for (int i = 0; i < this->rows + 1; ++i) {
        this->row_position[i] = mat.row_position[i];
    }
}

//move constructor
template<class T>
CSRMatrix<T>::CSRMatrix(CSRMatrix<T>&& mat)
    : Matrix<T>(mat.rows, mat.cols, false)
{

    this->rows = mat.rows;
    this->cols = mat.cols;
    this->nnzs = mat.nnzs;
    this->preallocated = true;
    if (this->preallocated) {
        this->values = mat.values;
        this->col_index = mat.col_index;
        this->row_position = mat.row_position;
        mat.values = nullptr;
        mat.col_index = nullptr;
        mat.row_position = nullptr;
    }  
}



// destructor
template <class T>
CSRMatrix<T>::~CSRMatrix()
{
    // Delete the values array
    if (this->preallocated) {
        delete[] this->row_position;
        delete[] this->col_index;
    }
    // The super destructor is called after we finish here
    // This will delete this->values if preallocated is true
}

// Explicitly print out the values in values array as if they are a matrix
template <class T>
void CSRMatrix<T>::printMatrix()
{
    std::cout << "Printing matrix" << std::endl;
    std::cout << "Values: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->values[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "row_position: ";
    for (int j = 0; j < this->rows + 1; j++)
    {
        std::cout << this->row_position[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "col_index: ";
    for (int j = 0; j < this->nnzs; j++)
    {
        std::cout << this->col_index[j] << " ";
    }
    std::cout << std::endl;
}

template<class T>
std::vector<T> CSRMatrix<T>::solve(std::vector<T>& b, std::vector<T>&x_init,
    std::string preconditioner, double tol, int max_iter) {

    CSRMatrix<T> precond{ util::sp::jacobi_precondition(*this) };
    
    std::vector<T>soln = util::sp::sparse_precond_cg(*this, x_init, b, tol
                , max_iter, precond);
   
    return soln;
}

template<class T>
Matrix<T> CSRMatrix<T>::solve(Matrix<T>& b,std::vector<T>& x_init,
    std::string preconditioner, double tol , int max_iter){

    //Matrix to store solution
    Matrix<T>x_matrix{ b.rows,b.cols };

    for (int k = 0; k < b.cols; ++k) {
        //split matrix b
        std::vector<T>b_slice = b.slice_col(k);

        //solve Ax = b for each b_slice
        bool decomposed = true;
        std::vector<T>x_vec = this->solve(b_slice, x_init,preconditioner,tol,max_iter);
        x_matrix.set_col(k, x_vec);
    }
    return x_matrix;
}

template<class T>
std::vector<int> CSRMatrix<T>::get_diags()
{
    int col_ctr = 0;
    //std::vector<T>diags(this->rows);
    std::vector<int>diags_idx(this->rows);
    for (int i = 0; i < this->rows; ++i) {

        int start = this->row_position[i];
        int end = this->row_position[i + 1];
        int init_ctr = col_ctr;
        bool has_diag = false;
        for (int j = start; j < end; ++j) {

           // if (this->col_index[col_ctr] > i - 1)break;
            if (i == this->col_index[col_ctr]) {
                has_diag = true;
                //diags[i] = this->values[col_ctr];
                diags_idx[i] = col_ctr;
                break;
            }
            ++col_ctr;

        }
        if (has_diag)col_ctr = init_ctr + end-start;
        if (!has_diag) {
            //diags[i] = 0;
            //-1: it's not on the non-zero values array;
            diags_idx[i] = -1;
        }   
    }
    return diags_idx;
}


// Do a matrix-vector product
// output = this * input
template<class T>
void CSRMatrix<T>::matVecMult(double* input, double* output)
{
    if (input == nullptr || output == nullptr)
    {
        std::cerr << "Input or output haven't been created" << std::endl;
        return;
    }

    // Set the output to zero
    for (int i = 0; i < this->rows; i++)
    {
        output[i] = 0.0;
    }

    int val_counter = 0;
    // Loop over each row
    for (int i = 0; i < this->rows; i++)
    {
        // Loop over all the entries in this col
        for (int val_index = this->row_position[i]; val_index < this->row_position[i + 1]; val_index++)
        {
            // This is an example of indirect addressing
            // Can make it harder for the compiler to vectorise!
            output[i] += this->values[val_index] * input[this->col_index[val_index]];

        }
    }
}

template<class T>
std::vector<T> CSRMatrix<T>::mat_vec_mult(std::vector<T> x)
{
    int n = this->rows;
    std::vector<T>y(n,0.0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = this->row_position[i]; j < this->row_position[i + 1]; ++j) {
            y[i] += this->values[j] * x[this->col_index[j]];
        }
    }
    return y;
}

template<class T>
CSRMatrix<T> CSRMatrix<T>::transpose()
{
    int c_rows = this->rows;
    int c_cols = this->cols;
    int c_nnzs = this->nnzs;
    int k = 0;
    int j= 0;
    int index = 0;
    CSRMatrix<T> csr{ c_rows, c_cols, c_nnzs, true };
    for (int i = 0; i < c_rows + 1; ++i)
        csr.row_position[i] = 0;

    std::vector<int> count(c_cols, 0);
    
    for (int i = 0; i < c_rows; ++i) {
        for (j = this->row_position[i]; j < row_position[i + 1]; ++j) {
            k = this->col_index[j];
            count[k]++;
        }
    }
    for (j = 0; j < c_cols; ++j) {
        csr.row_position[j + 1] = csr.row_position[j] + count[j];
    }
    for (j = 0; j < c_cols; ++j) {
        count[j] = 0;
    }
    for (int i = 0; i < c_rows; ++i) {
        for (int j = this->row_position[i]; j < row_position[i + 1]; ++j) {
            k = this->col_index[j];
            index = csr.row_position[k] + count[k];
            csr.col_index[index] = i;
            csr.values[index] = this->values[j];
            count[k]++;
        }
    }
    return csr;
}



// Do matrix matrix multiplication
// output = this * mat_right
template <class T>
void CSRMatrix<T>::matMatMult(CSRMatrix<T>& mat_right, CSRMatrix<T>& output)
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
        std::cerr << "OUTPUT HASN'T BEEN ALLOCATED" << std::endl;

    }

    // HOW DO WE SET THE SPARSITY OF OUR OUTPUT MATRIX HERE??
}