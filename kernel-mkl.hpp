#pragma once

#include <mkl.h>


// Aligned memory allocation for MKL.
struct MKLMemoryAllocator
{
	typedef double* TVector;

	static const int myAllocSize=64;

	static void allocate_vector   (size_t n, TVector &v)
	{ v = (double*) mkl_malloc(n*sizeof(double), myAllocSize); }

	static void deallocate_vector (TVector &v)
	{ delete v; }
};



// MKL-BLAS interface.
namespace mymkl {

    struct mvops {
    	static const int one = 1;
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            return ddot(&N, &x[0], &one, &y[0], &one);
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        {
            return ddot(&N, &x[0], &one, &x[0], &one);
        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            daxpy(&N, &alpha, &x[0], &one, &y[0], &one);
        }
    };

}

