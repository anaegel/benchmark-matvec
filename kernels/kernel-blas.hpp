#pragma once


#ifdef __APPLE__
#include <vecLib/vecLib.h>	// Apple CBLAS.
#include <Accelerate/Accelerate.h>
#else
#include <blas.h>
#endif



// Classic BLAS interface.
namespace myblas {

    struct mvops {
    	static const int one = 1;
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            return ddot_(&N, &x[0], &one, &y[0], &one);
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        {
            return dnorm2_(&N, &x[0], &one, &x[0], &one);
        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            daxpy_(&N, &alpha, &x[0], &one, &y[0], &one);
        }
    };

}

