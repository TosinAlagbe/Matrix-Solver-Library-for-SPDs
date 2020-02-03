
#ifndef SYMMETRIC_MATRIX_H
#define SYMMETRIC_MATRIX_H



#include "Matrix.h"
#include "SolverUtil.h"


//Symmetric Matrix Class
//Can be used to solve negative symmetric matrices as well as positive definte matrices
template<typename T>
class SymmetricMatrix : public Matrix<T> {

protected:
	//storing decomposed matrices
	DecompMatrix<T> decomp;
public:
	
	//default constructor
	SymmetricMatrix() {}

	// constructor where we want to preallocate ourselves
	//rows: row size of dense matrix
	//cols: col size of dense matrix
	//preallocate: boolean to denote allocation 
	SymmetricMatrix(int rows, int cols, bool preallocate = true);
	
	// constructor where we already have allocated memory outside
	//rows: row size of dense matrix
	//cols: col size of dense matrix
	//values_ptr: one-d array of dense matrix values traversed row-wise
	SymmetricMatrix(int rows, int cols, T* values_ptr);

	//solver: uses the LDLt factorization
	//b: Matrix of multiple right hand sides
	virtual Matrix<T> solve(Matrix<T>& b) override;
	//b: right hand side vector
	//decomposed: boolean to know if matrix is already decomposed
	virtual std::vector<T> solve(std::vector<T>& b, bool decomposed = false) override;
};

#endif // !SYMMETRIC_MATRIX_H