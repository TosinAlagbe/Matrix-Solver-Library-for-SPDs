
#ifndef SOLVER_UTIL_H
#define SOLVER_UTIL_H

#include <vector>
#include <memory>
#include <algorithm>


template<typename T>
class BandedMat;

template <class T>
class Matrix;

template<typename T>
class DecompMatrix;

template<typename T>
class Matrix1d;

template <class T>
class CSRMatrix;

//Solver Library containing different solvers, helper functions and factorization techniques
//Can be used as standalone functions
//For example: to use LU factorization for dense Matrix instead of the default GMRES:
//Call DecompMatrix<T> lu_factorize(const Matrix<T>& mat) on the dense matrix mat;
//Call std::vector<T> forward_sub(Matrix<T>& L,std::vector<T>& b) on the lower triangular matrix from lu_factorize
//where L is DecompMatrix<T>_.lower and b is the right hand side of the equation
//Then call std::vector<T>back_sub(Matrix<T>& u,std::vector<T>& y);
//where u is DecompMatrix<T>_.upper and y is the solution from forward_sub
//the solution from back_sub gives the solution to the equation

//Note: This library is made available if user doesn't want to use default solvers in the Matrix classes

namespace util {

	//L2 norm of vector vec
	template<typename T>
	T norm(std::vector<T>& vec);

	//Dense Matrix vector multiplication
	template <typename T>
	std::vector<T> matvecmult(Matrix<T>& A, std::vector<T>& x_vec);

	//Dense vector vector subtraction
	template<typename T>
	std::vector<double> vecvecsub(Matrix<T>& A, std::vector<T>& Ax_vec, std::vector<T>& b);

	namespace sp {

		//ilu(0) preconditioning
		template<typename T>
		CSRMatrix<T> ilu_precondition(CSRMatrix<T>& mat);

		//Jacobi preconditioning
		template<typename T>
		CSRMatrix<T> jacobi_precondition(CSRMatrix<T>& mat);

		//Inverse square root preconditioning
		template<typename T>
		CSRMatrix<T> inv_sqrt_precondition(CSRMatrix<T>& mat);

		//CSRMatrix  argument should be incompletely decomposed by a preconditioner
		template<typename T>
		std::vector<T> sparse_forward_sub(CSRMatrix<T>& mat, std::vector<T>& r);

		//CSRMatrix argument should be incompletely decomposed by a preconditioner
		template<typename T>
		std::vector<T> sparse_back_sub(CSRMatrix<T>& mat, std::vector<T>& r);

		//preconditioned conjugate gradient method
		template<typename T>
		std::vector<T> sparse_precond_cg(CSRMatrix<T>& mat, std::vector<T>& x_init, std::vector<T>& b_rhs,
			double tol, int iter, CSRMatrix<T>& precond);
		
		//conjugate gradient method
		template<typename T>
		std::vector<T> cg(CSRMatrix<T>& A_mat, std::vector<T>& b_vec, double tolerance = 1e-6, double iter_max = 1000000);

		template <typename T>
		T modul_sqr(T* A_matvec, int colA);

		//CSR Cholesky make use of  dense matrix to store and access the values of previous steps
		template<typename T>
		std::vector<double> CSR_cholesky(CSRMatrix <T>& sparse);
	}

	namespace band {
		//LU factorization for banded matrix 
		//Overwrites input matrix
		template<typename T>
		Matrix<T> band_gaussian(Matrix<T>* mat, int l_bw, int u_bw);

		//LU factorization with banded storage
		//Overwrites input matrix
		template<typename T>
		BandedMat<T> band_gaussian(BandedMat<T>* mat);

		//LLt factorization with banded storage
		//Overwrites input matrix
		template<typename T>
		void band_cholesky(BandedMat<T>* mat);

		//forward substitution for banded matrix
		template<typename T>
		std::vector<T> band_forward_sub(Matrix<T>& lower, std::vector<T>& b_rhs, int u_bw);

		//forward substitution with banded storage
		template<typename T>
		std::vector<T> band_forward_sub(BandedMat<T>& lower, std::vector<T>& b_rhs);

		//backward substitution for banded matrix
		template<typename T>
		std::vector<T> band_backward_sub(Matrix<T>& upper, std::vector<T>& b_rhs, int l_bw);

		//backward substitution with banded storage
		template<typename T>
		std::vector<T> band_backward_sub(BandedMat<T>& upper, std::vector<T>& b_rhs);

		//backward substitution with lower triangular matrix
		template<typename T>
		std::vector<T> band_back_sub_l(BandedMat<T>& lower, std::vector<T>& b);


	}
	namespace dense {
		//returns lower traingle and upper triangle matrix
		template<typename T>
		DecompMatrix<T> lu_factorize(const Matrix<T>& mat);

		//returns lower triangle matrix and diagonal vector
		template<typename T>
		DecompMatrix<T> ldlt_factorize(Matrix<T>* mat);

		//returns lower triangle matrix
		template<typename T>
		DecompMatrix<T> cholesky_factorize(Matrix<T>* mat);

		//general forward substitution
		template<typename T>
		std::vector<T> forward_sub(Matrix<T>& L,
			std::vector<T>& b);
		
		//general backward substitution
		template<typename T>
		std::vector<T>back_sub(Matrix<T>& u,
			std::vector<T>& y);

		//backward substitution with lower triangular matrix
		template<typename T>
		std::vector<T> back_sub_l(Matrix<T>& L,
			std::vector<T>& b);

		// GMRES solver for dense matrices
		template<typename T>
		std::vector<T> GMRES(Matrix<T>& A_mat, std::vector<T>& b_vec, double tolerance = 0.000001);

		// Aronold method for Hessenberg matrix
		template<typename T>
		void Arnold(Matrix<T>& A_mat, std::vector<T>& H_mat, 
				std::vector<T>& Q_ptr, int ite, int ite_max);

		// Given rotation method to solve the least square problem
		template<typename T>
		void rotation(std::vector<T>& H_mat, std::vector<T>& cosn,
					std::vector<T>& sinn, int ite, int ite_max);

		// matmatmult for GMRES
		template<typename T>
		std::vector<T>& mult(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA, int colB, std::vector<T>& Ar_mult);
		template<typename T>
		std::vector<T>& mult(T* A_matvec, std::vector<T>& B_matvec, int rowA, int colA, int colB, std::vector<T>& Ar_mult);


		//GMRES solver helper functions
		template<typename T>
		std::vector<T> sub(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA);
		template<typename T>
		std::vector<T>& sum(std::vector<T>& A_matvec, std::vector<T>& B_matvec, int rowA, int colA);
		template<typename T>
		T modul(std::vector<T>& A_matvec, int colA);
		

	}
	
}

#endif