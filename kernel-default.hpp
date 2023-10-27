#pragma once

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Klassische Routinen.
namespace classic {

    struct mvops {

        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            for (int i=0; i<N; ++i) { sum += x[i]*y[i]; }
            return sum;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            for (int i=0; i<N; ++i)  { y[i] = alpha*x[i] + y[i]; }
        }

    };
}


//! Nun mit SIMD.
namespace simd {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int i=0; i<N; ++i)
            { sum += x[i]*y[i];}
            return sum;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            #pragma omp simd
            for (int i=0; i<N; ++i)
            { y[i] = alpha*x[i] + y[i];}
        }
    };

}


//! Nun mit OpenMP+SIMD
namespace omp {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            #pragma omp parallel for simd shared(x,y,N) schedule(static) reduction(+:sum)
            for (int i=0; i<N; ++i)
            { sum += x[i]*y[i]; }
            return sum;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            #pragma omp parallel for simd shared(x,y,N) schedule(static)
            for (int i=0; i<N; ++i)
            { y[i] = alpha*x[i] + y[i];}
        }
    };

}


