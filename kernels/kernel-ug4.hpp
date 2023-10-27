#pragma once

#include "lib_algebra/cpu_algebra_types.h"

template <class TValue>
struct const_values
{
	constexpr static double FOUR = 4.0;
	constexpr static double ONE = 1.0;
	constexpr static double MINUS_ONE = -1.0;
};


template <typename TAlgebra>
struct UG4Allocator
{

	typedef typename TAlgebra::vector_type TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.create(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};



template <unsigned int bs>
struct UG4AllocatorBlock
{
	typedef typename ug::CPUBlockAlgebra<bs>::vector_type TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.create(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};


template <typename TAlgebra>
struct UG4AlgebraAllocator : public UG4Allocator<TAlgebra>
{
	typedef UG4Allocator<TAlgebra> base_type;
	typedef typename TAlgebra::vector_type TVector;
	typedef typename TAlgebra::matrix_type TMatrix;


	static size_t ijindex(size_t N, size_t i, size_t j)
	{ return i*N+j; }


	static TMatrix* create_matrix(size_t ncells)
	{
		const size_t nrows = ncells*ncells;
		TMatrix* mat = new TMatrix();
		mat->resize_and_clear(nrows, nrows);

		init_5pt_matrix(*mat, ncells); // std::cout << "init done!" << std::endl;

		return mat;
	}


	static void init_5pt_matrix(TMatrix& mat, int ncells)
	{
		const size_t N = ncells;
		const size_t n = mat.num_rows();
		assert(N*N == n);

		typedef typename TMatrix::value_type value_type;

		const value_type ONE = const_values<value_type>::ONE;
		const value_type DIAG = const_values<value_type>::FOUR;
		const value_type OFFD = const_values<value_type>::MINUS_ONE;

		// temporary storage
		typedef typename TMatrix::connection TConnection;
		const size_t NSTENCIL = 5;
		TConnection conn[NSTENCIL];


		// Fill:
		for (size_t j=0; j<N; ++j) // First "row", i=0
		{

			const size_t ind  = ijindex(N, 0, j);
			conn[0] = TConnection(ind, ONE);
			mat.set_matrix_row(ind, conn, 1);
		}

		for (size_t i=1; i<N-1; ++i)
		{
			{
				const size_t ind  = ijindex(N, i, 0);
				conn[0] = TConnection(ind, ONE);
				mat.set_matrix_row(ind, conn, 1);
			}

			for (size_t j=1; j<N-1; ++j)
			{
				const size_t ind = ijindex(N, i, j);

				conn[0] = TConnection(ind, DIAG);
				conn[1] = TConnection(ind-N, OFFD);
				conn[2] = TConnection(ind-1, OFFD);
				conn[3] = TConnection(ind+1, OFFD);
				conn[4] = TConnection(ind+N, OFFD);

				mat.set_matrix_row(ind, conn, 5);
			}

			{
				const size_t ind  = ijindex(N, i, N-1);
				conn[0] = TConnection(ind, ONE);
				mat.set_matrix_row(ind, conn, 1);
			}

		}

		for (size_t j=0; j<N; ++j )// Final "row" N-1
		{
			const size_t ind  = ijindex(N, N-1, j);
			conn[0] = TConnection(ind, ONE);
			mat.set_matrix_row(ind, conn, 1);
		}

		// Close: Copy to matrix.
		mat.defragment();
	}

};


// Define some allocators.
typedef UG4AlgebraAllocator<ug::CPUAlgebra> UG4AllocatorCPU1;
typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<2>> UG4AllocatorCPU2;

namespace myug4 {


struct mvops {

	// using  operations_vec.h
    template <class TVector>
    static double dot(const int N, const TVector &x, const TVector &y)
    { return ug::VecProd(x,y); }

    template <class TVector>
    static double norm2(const int N, const TVector &x)
    { return ug::VecNormSquared(x); } // ug::VecProd(x,x);

    template <class TVector>
    static void axpy(const int N, double alpha, const TVector &x, TVector &y)
    { ug::VecScaleAdd(y, alpha, x, 1.0, y); }


    template <class TVector, class TMatrix>
    static void matmul(const int N, TVector &y, const TMatrix &A, const TVector &x)
    { A.axpy(y, 0.0, y, 1.0, x); }

    template <class TVector, class TMatrix>
    static void matmul_transposed(const int N, TVector &y, const TMatrix &A, const TVector &x)
     { A.apply_transposed(y,x); }

    template <class TVector, class TMatrix>
    static void matmul_add(const int N, TVector &y, const TMatrix &A, const TVector &x)
    {  A.axpy(y, 1.0, y, 1.0, x);  }
};
}



// Some dummies.
namespace ug {

	LogAssistant& LogAssistant::instance() {}
}


void  ug_assert_failed() {}
