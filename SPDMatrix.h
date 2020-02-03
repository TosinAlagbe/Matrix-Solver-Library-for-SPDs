#ifndef SPD_MATRIX_H
#define SPD_MATRIX_H


#include "SymmetricMatrix.h"

//Symmetric Positive Definite Matrix Class
//Used to solve symmetric positive definite matrices only
//Uses the efficient Cholesky factorization
//Use this if your matrix is dense and symmetric poitive definite.
template<typename T>
class SPDMatrix :public SymmetricMatrix<T> {

public:

	//default constructor
	SPDMatrix() {}

	//constructor with internally preallocated values
	//rows: row size of dense matrix
	//cols: col size of dense matrix
	//preallocate: boolean to denote allocation 
	SPDMatrix(int rows, int cols, bool preallocate = true);	
	
	// constructor with externally preallocated values
	//rows: row size of dense matrix
	//cols: col size of dense matrix
	//values_ptr: boolean to denote allocation 
	SPDMatrix(int rows, int cols, T* values_ptr);

	//solver: uses the dense cholesky factorization
	//b: Matrix of multiple right hand sides
	Matrix<T> solve(Matrix<T>& b) override;
	//b: right hand side vector
	std::vector<T> solve(std::vector<T>& b, bool decomposed = false) override;
};

#endif // !SYMMETRIC_MATRIX_H

