#pragma once

#ifdef USE_MKL_BLAS

#include <mkl.h>


// Aligned memory allocation for MKL.
struct MKLMemoryAllocator
{
	typedef double* TVector;

	static const int myAllocSize=64;

	static void allocate_vector   (size_t n, TVector &v)
	{ v = (double*) mkl_malloc(n*sizeof(double), myAllocSize); }

	static void deallocate_vector (TVector &v)
	{ mkl_free(v); }
};



// MKL-BLAS interface.
namespace mymkl {

    struct mvops {
    	static const MKL_INT one;
        template <class TVector>
        static double dot(const int n, const TVector &x, const TVector &y)
        {
        	MKL_INT N=n;
            return ddot(&N, &x[0], &one, &y[0], &one);
        }

        template <class TVector>
        static double norm2(const int n, const TVector &x)
        {
        	MKL_INT N=n;
            return ddot(&N, &x[0], &one, &x[0], &one);
        }

        template <class TVector>
        static void axpy(const int n, double alpha, const TVector &x, TVector &y)
        {
        	MKL_INT N=n;
            daxpy(&N, &alpha, &x[0], &one, &y[0], &one);
        }
    };

}


#endif

void run_test_mkl(int niter, int c);