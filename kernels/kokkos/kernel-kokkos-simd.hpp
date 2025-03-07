#pragma once

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>


namespace mykokkos
{
    namespace simd {
        

        struct VectorAllocator
        {
            typedef Kokkos::Experimental::native_simd<double> simd_t;
            typedef Kokkos::View<simd_t *> TVector;

            static void allocate_vector(size_t n, TVector &p)
            {
                const size_t N = n / simd_t::size();
                p = TVector("x", N);
            }

            //! No dealloc (-> will be done when object is deleted...)
            static void deallocate_vector(TVector &p) {}
        };

    template <typename SIMD, typename T>
    KOKKOS_INLINE_FUNCTION T horizontal_add(const SIMD &v)
    {
        T sum = 0.0;
        for (int i = 0; i < SIMD::size(); i++)
        {
            sum += v[i]; // Sum each lane
        }
        return sum;
    };

    struct mvops
    {
        using simd_t=VectorAllocator::simd_t;
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            const size_t n = N / simd_t::size();
            double sum = 0.0;
            Kokkos::parallel_reduce("DOT", n, [=](int64_t i, double &lres)
                                    { lres += horizontal_add<simd_t, double>(x(i) * y(i)); }, sum);
            return sum;
           /* 
            KokkosVectorAllocator::simd_t agg(0.0);
            Kokkos::parallel_for("DOT", n, [=](const int64_t i)
                {  
                    agg+= x(i)*y(i) // agg=Kokkos::fma(x(i), y(i), agg);
                });
            return horizontal_add<KokkosVectorAllocator::simd_t, double>(agg);
            */
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x) 
        {
            const size_t n = N / simd_t::size();
            double sum = 0.0;
            Kokkos::parallel_reduce("NORM2", n, [=](int64_t i, double &lres)
                                   { lres += horizontal_add<simd_t, double>(x(i) * x(i)); }, sum);
           return sum;

           /* KokkosVectorAllocator::simd_t agg(0.0);
            Kokkos::parallel_for("NORM2", N, [=](const int64_t i)
                { agg+=x(i)*x(i);});
            return horizontal_add<KokkosVectorAllocator::simd_t, double>(agg);
            */

        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            Kokkos::parallel_for("DAXPY", N, [=](const int64_t i)
                                 { y(i) = alpha * x(i) + y(i); });
        }
    };

    void run_test(int niter, int c);

    }

}

#endif