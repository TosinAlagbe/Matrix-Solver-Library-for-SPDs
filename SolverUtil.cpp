#include "SolverUtil.h"
#include "CSRMatrix.h"

//LU Factorization
template<typename T>
DecompMatrix<T> util::dense::lu_factorize(const Matrix<T>& mat) {

    Matrix<T> temp{ mat };
    Matrix<T>lower{ mat.rows,mat.cols };

    //initialize lower matrix to zero
    lower.init_def(0);

    for (int k = 0; k < temp.rows - 1; ++k) {
        for (int i = k + 1; i < temp.rows; ++i) {

            T scale_factor = temp(i, k) / temp(k, k);
            for (int j = k; j < temp.rows; ++j) {
                temp(i, j) = temp(i, j) - scale_factor * temp(k, j);
            }
            lower(i, k) = scale_factor;
        }
    }

    //add the ones in the main diagonal of L
    //Matrix<T>eye =  diag(mat.rows, mat.cols, T(1)) ;
    lower = lower + diag(mat.rows, mat.cols, T(1));

    //store LU in a struct
    DecompMatrix<T> decomp{ lower,temp };
    return decomp;
}


//ldlt factorization
template<typename T>
DecompMatrix<T> util::dense::ldlt_factorize(Matrix<T>* mat) {

    //store L matrix for L and Lt
    Matrix<T> l_mat{ mat->rows, mat->cols };
    l_mat.init_def(0);
    l_mat = l_mat + diag(mat->rows, mat->cols, T(1));

    //store diagoanal D matrix vector
    Matrix1d<T> d_vec{ mat->rows };

    //store LD vector
    Matrix1d<T> v_vec{ mat->rows };
    v_vec.init_def(T(0));


    for (int i = 0; i < mat->rows; ++i) {
        for (int j = 0; j <= i - 1; ++j)
            v_vec(j) = l_mat(i, j) * d_vec(j);

        T sum = T(0);
        for (int k = 0; k <= i - 1; ++k)
            sum += l_mat(i, k) * v_vec(k);

        d_vec(i) = (*mat)(i, i) - sum;

        for (int j = i + 1; j < mat->rows; ++j) {

            T sum = 0;
            for (int k = 0; k <= i - 1; ++k)
                sum += l_mat(j, k) * v_vec(k);

            l_mat(j, i) = ((*mat)(j, i) - sum) / d_vec(i);
        }
    }
    DecompMatrix<T> decomp{ l_mat,d_vec };
    return decomp;
}

template<typename T>
DecompMatrix<T> util::dense::cholesky_factorize(Matrix<T>* mat) {

    Matrix<T> l_mat{ mat->rows, mat->cols };
    l_mat.init_def(0);
    //l_mat = l_mat + diag(mat.rows, mat.cols, T(1));

    l_mat(0, 0) = sqrt((*mat)(0, 0));

    int n = mat->rows;

    for (int j = 1; j < n; ++j)
        l_mat(j, 0) = (*mat)(j, 0) / l_mat(0, 0);

    for (int i = 1; i < n - 1; ++i) {

        T sum = 0;
        for (int k = 0; k <= i - 1; ++k) {
            sum += l_mat(i, k) * l_mat(i, k);
        }
        l_mat(i, i) = sqrt((*mat)(i, i) - sum);

        for (int j = i + 1; j < mat->rows; ++j) {
            T sum = 0;
            for (int k = 0; k <= i - 1; ++k) {
                sum += l_mat(j, k) * l_mat(i, k);
            }
            l_mat(j, i) = ((*mat)(j, i) - sum) / l_mat(i, i);
        }
    }
    //int n = mat.rows - 1;
    T sum = 0;
    for (int k = 0; k <= n - 1; ++k)
        sum += l_mat(n - 1, k) * l_mat(n - 1, k);

    l_mat(n - 1, n - 1) = sqrt((*mat)(n - 1, n - 1) - sum);

    DecompMatrix<T>decomp{ l_mat };

    return decomp;
}

template<typename T>
Matrix<T> util::band::band_gaussian(Matrix<T>* mat, int l_bw, int u_bw)
{
    int n = mat->cols;
    for (int k = 0; k < n - 1; ++k) {

        for (int i = k + 1; i <= std::min(k + l_bw, n-1); ++i)
            mat->operator()(i, k) = mat->operator()(i, k) 
                                / mat->operator()(k, k);
        
        for (int j = k + 1; j <= std::min(k + u_bw, n-1); ++j) {
            for (int i = k + 1; i <= std::min(k + l_bw, n-1); ++i)
                mat->operator()(i, j) = mat->operator()(i, j) 
                                - mat->operator()(i, k) 
                                * mat->operator()(k, j);
        }
    }
    mat->printMatrix();
    return *mat;
}
template<typename T>
BandedMat<T> util::band::band_gaussian(BandedMat<T>* mat) {

    int n = mat->col_sz;
    for (int k = 0; k < n - 1; ++k) {

        for (int i = k + 1; i <= std::min(k + mat->l_bw, n - 1); ++i)
            mat->operator()(i, k) = mat->operator()(i, k)
            / mat->operator()(k, k);

        for (int j = k + 1; j <= std::min(k + mat->u_bw, n - 1); ++j) {
            for (int i = k + 1; i <= std::min(k + mat->l_bw, n - 1); ++i)
                mat->operator()(i, j) = mat->operator()(i, j)
                - mat->operator()(i, k)
                * mat->operator()(k, j);
        }
    }
    return *mat;
}

template<typename T>
void util::band::band_cholesky(BandedMat<T>* mat){

    int n = mat->col_sz;
    int bw = mat->l_bw+ mat->u_bw+1 ;
    for (int j = 0; j < n; ++j) {
        int gamma = 0;
        for (int k = std::max(0, j - bw); k <= j - 1; ++k) {
            gamma = std::min(k + bw, n-1);
            for (int m = j; m <= gamma; ++m) {
                mat->operator()(m, j) =
                    mat->operator()(m, j) - mat->operator()(j, k)
                    * mat->operator()(m, k);
                std::cout << "(" << m << "," << j << "): " << mat->operator()(m, j) << "\n";
            }
        }
        gamma = std::min(j + bw, n-1);
        for (int m = j; m <= gamma; ++m) {
            mat->operator()(m, j) = mat->operator()(m, j)
                            / sqrt(mat->operator()(j, j));
            std::cout << "(" << m << "," << j << "): " << mat->operator()(m, j) << "\n";
        }
    }
    mat->printMatrix();

}

template<typename T>
std::vector<T> util::band::band_forward_sub(Matrix<T>& lower,
    std::vector<T>& b_rhs, int l_bw) {
    int n = lower.cols;
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i <= std::min(j + l_bw, n-1); ++i)
            b_rhs[i] = b_rhs[i] - lower(i, j) * b_rhs[j];
    }
    return b_rhs;
}

template<typename T>
std::vector<T> util::band::band_forward_sub(BandedMat<T>& lower, std::vector<T>& b_rhs) {
    int n = lower.col_sz;
    for (int j = 0; j < n; ++j) {
        for (int i = j + 1; i <= std::min(j + lower.l_bw, n - 1); ++i)
            b_rhs[i] = b_rhs[i] - lower(i, j) * b_rhs[j];
    }
    return b_rhs;
}


template<typename T>
std::vector<T> util::band::band_backward_sub(Matrix<T>& upper,
    std::vector<T>& b_rhs, int u_bw) {
    int n = upper.cols;
    for (int j = n - 1; j >= 0; --j) {
        b_rhs[j] = b_rhs[j] / upper(j, j);
        for (int i = std::max(0, j - u_bw); i <= j - 1; ++i)
            b_rhs[i] = b_rhs[i] - upper(i, j) * b_rhs[j];
    }
    return b_rhs;
}

template<typename T>
std::vector<T> util::band::band_backward_sub(BandedMat<T>& upper, std::vector<T>& b_rhs) {

    int n = upper.col_sz;
    for (int j = n - 1; j >= 0; --j) {
        b_rhs[j] = b_rhs[j] / upper(j, j);
        for (int i = std::max(0, j - upper.u_bw); i <= j - 1; ++i)
            b_rhs[i] = b_rhs[i] - upper(i, j) * b_rhs[j];
    }
    return b_rhs;
}

template<typename T>
std::vector<T> util::band::band_back_sub_l(BandedMat<T>& lower, std::vector<T>& b_rhs){
   
    int n = lower.col_sz;
    for (int j = n - 1; j >= 0; --j) {
        b_rhs[j] = b_rhs[j] / lower(j, j);
        for (int i = std::max(0, j - lower.u_bw); i <= j - 1; ++i)
            b_rhs[i] = b_rhs[i] - lower(j, i) * b_rhs[j];
    }
    return b_rhs;
}

template<typename T>
CSRMatrix<T> util::sp::ilu_precondition(CSRMatrix<T>& mat){
  
    int n = mat.rows;
    std::vector<int> counter(n,0);
    std::vector<int>diag = mat.get_diags();

    T* lu_values = new T[mat.nnzs];

    for (int i = 0; i < mat.nnzs; ++i)
        lu_values[i] = mat.values[i];


    for (int i = 1; i < n; ++i) {
        for (int v = mat.row_position[i] + 1; v < mat.row_position[i + 1];++v) {
            counter[mat.col_index[v]] = v;
        }
        for (int v = mat.row_position[i]; v < diag[i] ; ++v) {
            int j = mat.col_index[v];
            lu_values[v] = lu_values[v] / lu_values[diag[j]];
            for (int w = diag[j] + 1; w < mat.row_position[j + 1]; ++w) {
                int k = counter[mat.col_index[w]];
                if (k > 0)
                    lu_values[k] = lu_values[k] - lu_values[v] * lu_values[w];
            }
        }
        for (int v = mat.row_position[i] + 1; v < mat.row_position[i + 1]; ++v) {
            counter[mat.col_index[v]] = 0;
        }
    }
    
    int* lu_col_index = new int[mat.nnzs];
    int* lu_row_position = new int[mat.rows + 1];
    for (int i = 0; i < mat.nnzs; ++i) {
        lu_col_index[i] = mat.col_index[i];
    }
    for (int i = 0; i < mat.rows + 1; ++i) {
        lu_row_position[i] = mat.row_position[i];
    }
        
    CSRMatrix<T> ilu{ mat.rows,mat.cols,mat.nnzs, lu_values, lu_row_position, lu_col_index };
    return ilu;
}

template<typename T>
CSRMatrix<T> util::sp::jacobi_precondition(CSRMatrix<T>& mat){

    std::vector<int>diag_idx = mat.get_diags();
    CSRMatrix<T> jacobi_csr{ mat.rows,mat.cols,mat.rows,true };
    jacobi_csr.row_position[0] = 0;
    
    for (int i = 0; i < mat.rows; ++i) {
        
        if (diag_idx[i] != -1)
            jacobi_csr.values[i] = mat.values[diag_idx[i]];
        else
            jacobi_csr.values[i] = 0;

        jacobi_csr.col_index[i] = i;
        jacobi_csr.row_position[i + 1] = i + 1;
    }
    return jacobi_csr;
}

template<typename T>
CSRMatrix<T> util::sp::inv_sqrt_precondition(CSRMatrix<T>& mat){
    
    CSRMatrix<T>inv_sqrt_csr{ sp::jacobi_precondition(mat) };
    for (int i = 0; i < inv_sqrt_csr.rows; ++i) {
        inv_sqrt_csr.values[i] = 1/sqrt(abs(inv_sqrt_csr.values[i]));
    }
    return inv_sqrt_csr;
}

template<typename T>
std::vector<T> util::sp::sparse_forward_sub(CSRMatrix<T>& mat, std::vector<T>& r){

    int n = r.size();
    std::vector<int>diag = mat.get_diags();
    std::vector<T>y(n);
    y[0] = r[0];
    for (int i = 1; i < n; ++i) {
        int sum = 0;
        for (int j = mat.row_position[i]; j < diag[i]; ++j) {
            sum += mat.values[j] * y[mat.col_index[j]];
        }
        y[i] = r[i] - sum;
    }
    return y;
}

template<typename T>
std::vector<T> util::sp::sparse_back_sub(CSRMatrix<T>& mat, std::vector<T>& y){
   
    int n = y.size();
    std::vector<int>diags = mat.get_diags();
    std::vector<T>z(n);
    z[n - 1] = y[n - 1] / mat.values[diags[n - 1]];
    for (int i = n - 2; i >= 0; --i) {
        int sum = 0;
        for (int j = diags[i]; j < mat.row_position[i + 1]; ++j) 
            sum += mat.values[j] * z[mat.col_index[j]];
        z[i] = (y[i] - sum) / mat.values[diags[i]];
    }
    return z;
}

template<typename T>
std::vector<T> util::sp::sparse_precond_cg(CSRMatrix<T>& mat, std::vector<T>&x_init, std::vector<T>& b_rhs,
    double tol, int max_iter,CSRMatrix<T>& precond)
{
    int n = b_rhs.size();
    std::vector<T> res(n);
   // Ax is vector
    std::vector<T> Ax = mat.mat_vec_mult(x_init);
    std::transform(b_rhs.begin(), b_rhs.end(), Ax.begin(), res.begin(), std::minus<T>());
    
    //solve w = (C^-1)r where C is preconditioner
    std::vector<T>xx = util::sp::sparse_forward_sub(precond, res);
    std::vector<T>w = util::sp::sparse_back_sub(precond, xx);

    //solve v = ((C^t)^-1)w
    CSRMatrix<T>precond_t{ precond.transpose() };
    std::vector<T>pp = util::sp::sparse_forward_sub(precond_t, w);
    std::vector<T>v = util::sp::sparse_back_sub(precond_t, pp);

    std::vector<T>tv(n);
    std::vector<T>tu(n);
    std::vector<T>sv(n);
    
    double alpha = 0;
    for (int i = 0; i < n; ++i) {
        alpha += w[i] * w[i];
    }
    int k = 1;
    while (k <= max_iter) {

        if (norm(v) < tol) {
            std::cout << "\n Solved in " << k << " iterations!\n";
            return x_init;
        }

        std::vector<T>u = mat.mat_vec_mult(v);
        double vu_sum = 0;
        for (int i = 0; i < n; ++i) {
            vu_sum += v[i] * u[i];
        }
        double t = alpha / vu_sum;
        
        //tv = t*v
        std::transform(v.begin(), v.end(), tv.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, t));
        //x = x+tv
        std::transform(x_init.begin(), x_init.end(), tv.begin(), x_init.begin(), std::plus<T>());

        
        //tu = t*u
        std::transform(u.begin(), u.end(), tu.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, t));
        //res = res - tu
        std::transform(res.begin(), res.end(), tu.begin(), res.begin(), std::minus<T>());

        //solve w = (C^-1)r where C is preconditioner
        xx = util::sp::sparse_forward_sub(precond, res);
        w = util::sp::sparse_back_sub(precond, xx);

        double beta = 0;
        for (int i = 0; i < n; ++i) {
            beta += w[i] * w[i];
        }
        if (abs(beta) < tol) {
            if (norm(res) < tol) {
                std::cout << "\n Solved in " << k << " iterations!\n";
                return x_init;
            }        
        }
        double s = beta / alpha;
        
        //solve v = ((C^t)^-1)w
       // CSRMatrix<T>precond_t{ precond.transpose() };
        pp = util::sp::sparse_forward_sub(precond_t, w);
        std::vector<T>c_tw = util::sp::sparse_back_sub(precond_t, pp);
          
        //sv = s* v;
        std::transform(v.begin(), v.end(), sv.begin(),
            std::bind(std::multiplies<T>(), std::placeholders::_1, s));

        //v = c_tw+ sv
        std::transform(c_tw.begin(), c_tw.end(), sv.begin(), v.begin(), std::plus<T>());

        alpha = beta;

        k = k + 1;  
    }
    if (k > max_iter){
        std::cerr << "\nMax no. of iterations was exceeded\n";
        return std::vector<T>();
    }  
}

//L2 norm of vector
template<typename T>
T util::norm(std::vector<T>& vec){

    T sum = 0;
    for (int i = 0; i < vec.size(); ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}


    



//lower must be lower triangular matrix
//lower must be square and same row length as b_rhs
//
template<typename T>
std::vector<T> util::dense::forward_sub(Matrix<T>& lower,
    std::vector<T>& b_rhs) {

    std::vector<T> y(b_rhs.size());
    y[0] = b_rhs[0] / lower(0, 0);
    for (int i = 1; i < b_rhs.size(); ++i) {
        T sum = 0;
        for (int j = 0; j <= i - 1; ++j) {
            sum += lower(i, j) * y[j];
        }
        y[i] = (b_rhs[i] - sum) / lower(i, i);
    }
    return y;
}

template<typename T>
std::vector<T> util::dense::back_sub_l(Matrix<T>& l,
    std::vector<T>& y) {
    int n = y.size();
    std::vector<T> x(n);
    x[n - 1] = y[n - 1] / l(n - 1, n - 1);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (int j = i + 1; j < n; ++j) 
            sum += l(j, i) * x[j];
        x[i] = (y[i] - sum) / l(i, i);
    }
    return x;
}


template<typename T>
std::vector<T> util::dense::back_sub(Matrix<T>& u,
    std::vector<T>& y) {
    int n = y.size();
    std::vector<T> x(n);
    x[n - 1] = y[n - 1] / u(n - 1, n - 1);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0;
        for (int j = i + 1; j < n; ++j)
            sum += u(i, j) * x[j];
        x[i] = (y[i] - sum) / u(i, i);
    }
    return x;
}

//template<typename T>
//std::vector<T> util::matvecmult(Matrix<T>& A, std::vector<T>& x_vec)
//{
//    std::vector<T> Ax_vec;
//    Ax_vec.resize(A.rows);
//    for (int i = 0; i < A.rows; i++)
//    {
//        Ax_vec[i] = 0;
//    }
//    for (int i = 0; i < A.rows; i++)
//    {
//        for (int j = 0; j < A.cols; j++)
//        {
//            Ax_vec[i] += A.values[i * A.cols + j] * x_vec[j];
//        }
//    }
//    return Ax_vec;
//}

template <class T>
std::vector<T> util::sp::cg(CSRMatrix<T>& A_mat, std::vector<T>& b_vec, double tolerance, double iter_max)
{
    // unique pointers of matrices and vectors
    std::vector<T> x_vec(A_mat.rows);
    std::unique_ptr<T[]> Ax_mult(new T[A_mat.rows]);
    std::unique_ptr<T[]> Ar_mult(new T[A_mat.rows]);
    std::unique_ptr<T[]> residual(new T[A_mat.rows]);
    std::unique_ptr<T[]> rho(new T[A_mat.rows]);
    std::unique_ptr<T> rAr_mult(new T);
    rAr_mult.get()[0] = 0;
    // parameters
    int iter = 0;
    double norm_res_o = 0;
    double norm_res_n = 1;

    // define initial guess of vector to be zero
    for (int i = 0; i < A_mat.rows; i++)
    {
        x_vec[i] = 0;
        Ax_mult.get()[i] = 0;
        Ar_mult.get()[i] = 0;
        residual.get()[i] = 0;
        rho.get()[i] = 0;

    }

    // matvecmult of matrix A and vector x
    for (int i = 0; i < A_mat.rows; i++)
    {
        for (int val_index = A_mat.row_position[i]; val_index < A_mat.row_position[i + 1]; val_index++)
        {
            Ax_mult.get()[i] += A_mat.values[val_index] * x_vec[A_mat.col_index[val_index]];
        }
    }

    // calculating the difference between Ax and b where Ax = b
    for (int i = 0; i < A_mat.cols; i++)
    {
        residual.get()[i] = b_vec[i] - Ax_mult.get()[i];
    }

    // modulus squared of the residual vector
    for (int i = 0; i < A_mat.cols; i++)
    {
        norm_res_o += residual.get()[i] * residual.get()[i];
    }

    for (int i = 0; i < A_mat.rows; i++)
    {
        rho.get()[i] = residual.get()[i];
    }

    // conjugate gradient loop for solving Ax=b
    iter = 0;
    while (tolerance < sqrt(norm_res_n) && iter < iter_max)
    {
        // matvecmult of matrix A and vector rho
        for (int i = 0; i < A_mat.rows; i++)
        {
            for (int val_index = A_mat.row_position[i]; val_index < A_mat.row_position[i + 1]; val_index++)
            {
                Ar_mult.get()[0] += A_mat.values[val_index] * rho.get()[A_mat.col_index[val_index]];
            }
        }

        // vecvecmult of vector r and vector r
        for (int i = 0; i < A_mat.cols; i++)
        {
            rAr_mult.get()[i] += rho.get()[i] * Ar_mult.get()[i];
        }


        T alpha = norm_res_o / (*rAr_mult.get());

        for (int i = 0; i < A_mat.rows; i++)
        {
            x_vec[i] += alpha * rho.get()[i];
            residual.get()[i] -= alpha * Ar_mult.get()[i];
        }

        norm_res_n = modul_sqr(residual.get(), A_mat.rows);

        for (int i = 0; i < A_mat.rows; i++)
        {
            rho.get()[i] = residual.get()[i] + (norm_res_n / norm_res_o) * rho.get()[i];
        }

        norm_res_o = norm_res_n;
        ++iter;
    }
  
    return x_vec;
}

template <typename T>
T util::sp::modul_sqr(T* A_matvec, int colA)
{
    T result = 0;
    for (int i = 0; i < colA; i++)
    {
        result += A_matvec[i] * A_matvec[i];
    }
    return result;
}

template <typename T>
std::vector<double> util::sp::CSR_cholesky(CSRMatrix <T>& sparse)
{
    T sum_over_row_sq = 0;
    T temp_diag = 0;
    int num_row = 0;
    T sum_mult = 0;
    T non_diag;
    T diag;
    T val;
    int j = 0;
    int temp_index;

    std::vector <T> l_1d;

    std::vector<T> temp(sparse.cols);
    std::vector<std::vector<T>> l(sparse.rows, temp);


    l.reserve(sparse.nnzs * 5);

    for (int i = 0; i < sparse.rows; i++)
    {
        if (sparse.row_position[i + 1] - sparse.row_position[i] != 0)
        {
            num_row = sparse.row_position[i + 1] - sparse.row_position[i];
            sum_over_row_sq = 0;
        }
        for (int k = 0; k < sparse.cols; k++)
        {
            if (k == sparse.col_index[j])
            {
                val = sparse.values[j];
                j += 1;
            }
            else
            {
                val = 0;
            }
            //Cholesky Decomposition
            if (k == i)
            {
                diag = std::pow((val - sum_over_row_sq), 0.5);
                l[i][k] = diag;
                l_1d.push_back(diag);
            }
            else if (k < i)
            {
                sum_mult = 0;
                for (int z = 0; z < k; z++)
                {
                    temp_index = sparse.col_index[j];
                    sum_mult += (l[i][z]) * (l[k][z]);
                }
                non_diag = (1 / l[k][k]) * (val - sum_mult);
                if (non_diag != 0)
                {
                    l[i][k] = non_diag;
                    l_1d.push_back(non_diag);
                }
                sum_over_row_sq += std::pow(non_diag, 2);
            }
            else
                continue;
        }
    }
    return l_1d;
}

template <typename T>
std::vector<double> util::vecvecsub(Matrix<T>& A, std::vector<T>& Ax_vec, std::vector<T>& b)
{
    std::vector<double> residual;
    residual.resize(A.cols);
    for (int i = 0; i < A.cols; i++)
    {
        residual[i] = 0;
    }

    for (int i = 0; i < A.cols; i++)
    {
        residual[i] = Ax_vec[i] - b[i];
    }
    return residual;
}


template<typename T>
std::vector<T> util::matvecmult(Matrix<T>& A, std::vector<T>& x_vec)
{
    std::vector<T> Ax_vec;
    Ax_vec.resize(A.rows);
    for (int i = 0; i < A.rows; i++)
    {
        Ax_vec[i] = 0;
    }
    for (int i = 0; i < A.rows; i++)
    {
        for (int j = 0; j < A.cols; j++)
        {
            Ax_vec[i] += A.values[i * A.cols + j] * x_vec[j];
        }
    }
    return Ax_vec;
}



template <class T>
std::vector<T> util::dense::GMRES(Matrix<T>& A_mat, std::vector<T>& b_vec, double tolerance)
{
    int ite_max = std::max(A_mat.rows * A_mat.cols,1000);
    std::vector<T> x(A_mat.rows);
    std::vector<T>Q_ptr(A_mat.rows * (ite_max + 1));
    std::vector<T>Ax_vec(A_mat.rows);
    std::vector<T>sinn(ite_max+1);
    std::vector<T>cosn(ite_max+1);
    std::vector<T>res_norm_vec(1 + ite_max);
    std::vector<T>Qy_mult(A_mat.rows);
    std::vector<T>H_mat((1 + ite_max) * (1 + ite_max));
    std::vector<T>y_vec(ite_max);
    std::vector<T>Q_mat(A_mat.rows * ite_max);
  
    T b_vec_norm = util::dense::modul(b_vec, A_mat.cols);

    for (int i = 0; i < (1 + ite_max); i++)
    {
        sinn[i] = 0;
        cosn[i] = 0;
        for (int j = 0; j < (1 + ite_max); j++)
        {
            H_mat[i * (1 + ite_max) + j] = 0;
        }
    }

    for (int i = 0; i < A_mat.rows; i++)
    {
        x[i] = 0;
    }

    util::dense::mult(A_mat.values, x, A_mat.rows, A_mat.cols, 1, Ax_vec);
    std::vector<T> residual = sub(Ax_vec, b_vec, A_mat.cols, 1);
    T modulus = modul(residual, A_mat.cols);

    res_norm_vec[0] = modulus;

    for (int i = 0; i < A_mat.cols; i++)
    {
        Q_ptr[i * (ite_max + 1)] = residual[i] / modulus;
    }

    int ite = 0;
    double err = 1;
    while (ite < ite_max && err > tolerance)
    {
        // arnold
        util::dense::Arnold(A_mat, H_mat, Q_ptr, ite, ite_max);
        // rotation
        util::dense::rotation(H_mat, cosn, sinn, ite, ite_max);
        // residual vector
        res_norm_vec[ite + 1] = -sinn[ite] * res_norm_vec[ite];
        res_norm_vec[ite] = -cosn[ite] * res_norm_vec[ite];
        err = abs(res_norm_vec[ite + 1]) / b_vec_norm;
        ++ite;
    }

    for (int i = ite - 1; i >= 0; i--)
    {
        for (int j = ite - 1; j > i; j--)
        {
            res_norm_vec[i] -= H_mat[i * (ite_max + 1) + j] * y_vec[j];
        }
        y_vec[i] = res_norm_vec[i] / H_mat[i * (ite_max + 1) + i];
    }

    for (int i = 0; i < A_mat.rows; i++)
    {
        for (int j = 0; j < ite; j++)
        {
            Q_mat[i * ite_max + j] = Q_ptr[i * (ite_max + 1) + j];
        }
    }
    util::dense::mult(Q_mat, y_vec, A_mat.rows, ite_max, 1, Qy_mult);

    for (int i = 0; i < A_mat.rows; i++)
    {
        x[i] += Qy_mult[i];
    }

    return x;
}

template<typename T>
void util::dense::Arnold(Matrix<T>&A_mat, std::vector<T>& H_mat, std::vector<T>& Q_ptr, int ite, int ite_max)
{
    for (int i = 0; i < A_mat.rows; i++)
    {
        for (int j = 0; j < A_mat.cols; j++)
        {
            Q_ptr[i * (ite_max + 1) + ite + 1] += A_mat.values[i * A_mat.cols + j] * Q_ptr[j * (ite_max + 1) + ite];
        }
    }
    for (int i = 0; i <= ite; i++)
    {
        for (int j = 0; j < A_mat.rows; j++)
        {
            H_mat[i * (ite_max + 1) + ite] += Q_ptr[j * (ite_max + 1) + ite + 1] * Q_ptr[j * (ite_max + 1) + i];
        }
        for (int j = 0; j < A_mat.rows; j++)
        {
            Q_ptr[j * (ite_max + 1) + ite + 1] -= H_mat[i * (ite_max + 1) + ite] * Q_ptr[j * (ite_max + 1) + i];
        }
    }
    H_mat[(ite + 1) * (ite_max + 1) + ite] = 0;
    for (int i = 0; i < A_mat.rows; i++)
    {
        H_mat[(ite + 1) * (ite_max + 1) + ite] += Q_ptr[i * (ite_max + 1) + ite + 1] * Q_ptr[i * (ite_max + 1) + ite + 1];
    }
    H_mat[(ite + 1) * (ite_max + 1) + ite] = sqrt(H_mat[(ite + 1) * (ite_max + 1) + ite]);

    for (int i = 0; i < A_mat.rows; i++)
    {
        Q_ptr[i * (ite_max + 1) + ite + 1] = Q_ptr[i * (ite_max + 1) + ite + 1] / H_mat[(ite + 1) * (ite_max + 1) + ite];
    }
}

template<typename T>
void util::dense::rotation(std::vector<T>& H_mat, std::vector<T>& cosn, 
    std::vector<T>& sinn, int ite, int ite_max)
{
    for (int i = 0; i <= ite - 1; i++)
    {
        T temp = cosn[i] * H_mat[i * (ite_max + 1) + ite] + sinn[i] * H_mat[(i + 1) * (ite_max + 1) + ite];
        H_mat[(i + 1) * (ite_max + 1) + ite] = -sinn[i] * H_mat[i * (ite_max + 1) + ite] + cosn[i] *
            H_mat[(i + 1) * (ite_max + 1) + ite];
        H_mat[i * (ite_max + 1) + ite] = temp;
    }

    if (H_mat[ite * (ite_max + 1) + ite] != 0)
    {
        T temp = sqrt(H_mat[ite * (ite_max + 1) + ite] * H_mat[ite * (ite_max + 1) + ite] +
            H_mat[(ite + 1) * (ite_max + 1) + ite] * H_mat[(ite + 1) * (ite_max + 1) + ite]);
        cosn[ite] = abs(H_mat[ite * (ite_max + 1) + ite]) / temp;
        sinn[ite] = cosn[ite] * H_mat[(ite + 1) * (ite_max + 1) + ite] / H_mat[ite * (ite_max + 1) + ite];
    }
    else
    {
        cosn[ite] = 0;
        sinn[ite] = 1;
    }
    H_mat[ite * (ite_max + 1) + ite] = cosn[ite] * H_mat[ite * (ite_max + 1) + ite] +
        sinn[ite] * H_mat[(ite + 1) * (ite_max + 1) + ite];
    H_mat[(ite + 1) * (ite_max + 1) + ite] = 0;
}


template <class T>
std::vector<T>& util::dense::mult(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA, int colB, std::vector<T>& Ar_mult)
{
    for (int i = 0; i < rowA; i++)
    {
        Ar_mult[i] = 0;
    }
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            for (int k = 0; k < colA; k++)
            {
                Ar_mult[i * colB + j] += A_matvec[i * colA + k] * B_matvec[k * colB + j];
            }
        }
    }
    return Ar_mult;
}

template <class T>
std::vector<T>& util::dense::mult(T* A_matvec, std::vector<T>& B_matvec, int rowA, int colA, int colB, std::vector<T>& Ar_mult)
{
    for (int i = 0; i < rowA; i++)
    {
        Ar_mult[i] = 0;
    }
    for (int i = 0; i < rowA; i++)
    {
        for (int j = 0; j < colB; j++)
        {
            for (int k = 0; k < colA; k++)
            {
                Ar_mult[i * colB + j] += A_matvec[i * colA + k] * B_matvec[k * colB + j];
            }
        }
    }
    return Ar_mult;
}


template <class T>
std::vector<T> util::dense::sub(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA)
{
    std::vector<T>result(rowA * colA);
   
    for (int i = 0; i < rowA * colA; i++)
    {
        result[i] = A_matvec[i] - B_matvec[i];
    }
    return result;
}

template <class T>
std::vector<T>& util::dense::sum(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA)
{
    std::vector<T>& result(rowA * colA);
    
    for (int i = 0; i < rowA * colA; i++)
    {
        result[i] = A_matvec[i] + B_matvec[i];
    }
    return result;
}

template <class T>
T util::dense::modul(std::vector<T>& A_matvec, int colA)
{
    T result = 0;
    for (int i = 0; i < colA; i++)
    {
        result += A_matvec[i] * A_matvec[i];
    }
    result = sqrt(result);
    return result;
}





