#pragma once


#include <Eigen/Sparse>

struct EigenVectorAllocator
{
	typedef Eigen::VectorXd TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.resize(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }

};



struct EigenMatrixVectorAllocator : public EigenVectorAllocator
{
	typedef EigenVectorAllocator::TVector TVector;
	typedef Eigen::SparseMatrix<double> TMatrix;

	static TMatrix* create_matrix(size_t ncells)
	{
		const size_t nrows = ncells*ncells;
		TMatrix* mat = new TMatrix(nrows,nrows);

		mat->reserve(Eigen::VectorXi::Constant(nrows,6)); // std::cout << "reserve done!" << std::endl;
		init_5pt_matrix(*mat, ncells); // std::cout << "init done!" << std::endl;

		return mat;
	}


	static size_t ijindex(size_t N, size_t i, size_t j)
	{ return i*N+j; }


	static void init_5pt_matrix(TMatrix& mat, int N)
	{
		const size_t n = mat.rows();

		// Init: Create triplets.
		typedef Eigen::Triplet<double> T;
		std::vector<T> tripletList;
		tripletList.reserve(n*5);


		// Fill:
		for (size_t j=0; j<N; ++j)
		{
			// First "row", i=0
			const size_t ind  = ijindex(N, 0, j);
			tripletList.push_back(T(ind,ind, 1.0));
		}

		for (size_t i=1; i<N-1; ++i)
		{
			{
				const size_t ind  = ijindex(N, i, 0);
				tripletList.push_back(T(ind,ind, 1.0));
			}

			for (size_t j=1; j<N-1; ++j)
			{
				const size_t ind = ijindex(N, i, j);
				tripletList.push_back(T(ind,ind-N, -1.0));
				tripletList.push_back(T(ind,ind-1, -1.0));
				tripletList.push_back(T(ind,ind,	4.0));
				tripletList.push_back(T(ind,ind+1, -1.0));
				tripletList.push_back(T(ind,ind+N, -1.0));
			}

			{
				const size_t ind  = ijindex(N, i, N-1);
				tripletList.push_back(T(ind,ind, 1.0));
			}

		}

		for (size_t j=0; j<N; ++j)
		{
			const size_t ind = ijindex(N, N-1, j);
			tripletList.push_back(T(ind,ind, 1.0));  // set_matrix_row
		}

		// DEBUG
		// for (auto e : tripletList)
		// { std::cout << e.row() << "," << e.col() << ":" <<e.value()  <<std::endl; }

		// Close: Copy to matrix.
		mat.setFromTriplets(tripletList.begin(), tripletList.end());
	}

};


namespace myeigen {
struct mvops {

    template <class TVector>
    static double dot(const int N, const TVector &x, const TVector &y)
    { return x.dot(y); }

    template <class TVector>
    static double norm2(const int N, const TVector &x)
    { return x.dot(x); }

    template <class TVector>
    static void axpy(const int N, double alpha, const TVector &x, TVector &y)
    { y += alpha*x; }

    template <class TVector, class TMatrix>
    static void matmul_set(const int N, TVector &y, const TMatrix &A, const TVector &x)
    { y = A*x; }

    template <class TVector, class TMatrix>
    static void matmul_add(const int N, TVector &y, const TMatrix &A, const TVector &x)
    { y += A*x; }
};
}
