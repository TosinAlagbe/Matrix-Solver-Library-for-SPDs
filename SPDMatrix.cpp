#include "SPDMatrix.h"

template<typename T>
SPDMatrix<T>::SPDMatrix(int rows, int cols, bool preallocate)
	:SymmetricMatrix<T>{ rows, cols, preallocate } {}

template<typename T>
SPDMatrix<T>::SPDMatrix(int rows, int cols, T* values_ptr)
	: SymmetricMatrix<T>{ rows,cols, values_ptr } {}


template<typename T>
Matrix<T> SPDMatrix<T>::solve(Matrix<T>& b)
{
	//Matrix to store solution
	Matrix<T>x_matrix{ b.rows,b.cols };

	//decompose to llt (cholesky)
	this->decomp = util::dense::cholesky_factorize(this);

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

template<typename T>
std::vector<T> SPDMatrix<T>::solve(std::vector<T>& b, bool decomposed )  {

	if (!decomposed) {
		//decompose to llt
		this->decomp = util::dense::cholesky_factorize(this);
	}
	int n = this->rows;
	//solve Ly = b
	std::vector<T>y = util::dense::forward_sub(this->decomp.lower, b);
	//solve Lt x = y 
	std::vector<T>x = util::dense::back_sub_l(this->decomp.lower, y);
	return x;

}

