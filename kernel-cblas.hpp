#pragma once

#ifdef __APPLE__
#include <vecLib/vecLib.h>	// Apple CBLAS.
#else
#include <cblas.h>
#endif


#ifdef USE_CBLAS
// CBLAS
namespace mycblas {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        { return cblas_ddot(N, &x[0], 1, &y[0], 1); }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return cblas_dnrm2(N, &x[0], 1); }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        { cblas_daxpy(N, alpha, &x[0], 1, &y[0], 1); }
    };

}
#endif
