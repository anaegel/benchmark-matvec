#pragma once

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>

// Klassische Routinen.
namespace mykokkos
{
    namespace plain {
    
    struct VectorAllocator
    {
            using TVector = Kokkos::View<double*>;
           
            static void allocate_vector(size_t n, TVector &p) 
            {
                p = TVector( Kokkos::view_alloc(Kokkos::WithoutInitializing), n);
            }

            static void deallocate_vector(TVector &p) 
            {
                Kokkos::fence();  // Ensure all kernels touching this view finished
                p = {};
            }
        };

    struct mvops
    {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            Kokkos::parallel_reduce("DOT", N, KOKKOS_LAMBDA (int64_t i, double &lres)
            { lres += x[i] * y[i]; }, sum);
            return sum;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x) 
        {
            double sum = 0.0;
            Kokkos::parallel_reduce("NORM2", N, KOKKOS_LAMBDA (int64_t i, double &lres)
            { lres += x[i] * x[i]; }, sum);
           return sum;
        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            Kokkos::parallel_for("DAXPY", N, KOKKOS_LAMBDA (const int64_t i)
            { y[i] = alpha * x[i] + y[i]; });
        }
    };

    void run_test(int niter, int c);
    }
}

#endif