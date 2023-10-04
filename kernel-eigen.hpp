#pragma once


#include <Eigen/Sparse>

struct EigenVectorAllocator
{
	typedef Eigen::VectorXd TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.resize(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }




	typedef Eigen::SparseMatrix<double> TMatrix;
	static TMatrix* create_matrix(size_t n)
	{
		const size_t nrows = n*n;
		TMatrix* mat = new TMatrix(nrows,nrows);
		mat->reserve(Eigen::VectorXi::Constant(nrows,6));
		return mat;
	}

protected:
	static size_t ijindex(size_t N, size_t i, size_t j)
	{ return i*N+j; }

public:
	static void init_5pt_matrix(TMatrix& mat, int N)
	{
		const size_t n = mat.rows();

		// Create triplets.
		typedef Eigen::Triplet<double> T;
		std::vector<T> tripletList;
		tripletList.reserve(n*5);


		for (size_t j=0; j<N; ++j)
		{
			const size_t ind  = ijindex(N, 0, j);
			tripletList.push_back(T(ind,ind, 1.0));
		}

		for (size_t i=0; i<N; ++i)
		{
			{
				const size_t ind  = ijindex(N, i, 0);
				tripletList.push_back(T(ind,ind, 1.0));
			}

			for (size_t j=1; j<n-1; ++j)
			{
				const size_t ind = ijindex(N, i, j);
				tripletList.push_back(T(ind,ind-n, -1.0));
				tripletList.push_back(T(ind,ind-1, -1.0));
				tripletList.push_back(T(ind,ind,	4.0));
				tripletList.push_back(T(ind,ind+1, -1.0));
				tripletList.push_back(T(ind,ind+n, -1.0));
			}

			{
				const size_t ind  = ijindex(N, i, n-1);
				tripletList.push_back(T(ind,ind, 1.0));
			}

		}

		for (size_t j=0; j<n; ++j)
		{
			const size_t ind = ijindex(N, n-1, j);
			tripletList.push_back(T(ind,ind, 1.0));
		}

		// Fill matrix.
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
    static void matmul(const int N, TMatrix &A, const TVector &x)
    {}
};
}
