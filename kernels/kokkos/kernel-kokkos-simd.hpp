#pragma once

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>


namespace mykokkos
{
    namespace simd {
        
        struct VectorAllocator
        {
            using simd_t = Kokkos::Experimental::simd<double>;
            using TVector = Kokkos::View<simd_t*, Kokkos::LayoutRight>;

            static void allocate_vector(size_t n, TVector &p) {

                const size_t Npacks = (n + simd_t::size() - 1) / simd_t::size(); // rundet hoch, falls Tail
                p = TVector( Kokkos::view_alloc(Kokkos::WithoutInitializing), Npacks );
/*
#ifndef NDEBUG
              std::cerr << " simd_t::size() = "<<  simd_t::size() << std::endl;
                // Debug: print pointer and alignment modulo 64 bytes
                auto ptr = reinterpret_cast<uintptr_t>(p.data());
                fprintf(stderr, "[debug] allocate_vector: view ptr=%p mod64=%zu packs=%zu\n", (void*)ptr, (size_t)(ptr % 64), Npacks);
#endif
*/
            }

        
            //! Dealloc (-> will be done when object is deleted...)
            static void deallocate_vector(TVector &p) {
                // Ensure all kernels touching this view finished
                Kokkos::fence();
                // Release the managed View (dekrementiert Reference und gibt Speicher frei)
                p = TVector(); // or: p = {};
            }
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

            // Preferred: perform the accumulation in simd lanes and do
            // the horizontal add only once at the end. This avoids the
            // per-iteration lane-sum overhead.
            simd_t agg(0.0);
            Kokkos::parallel_reduce("DOT", n, KOKKOS_LAMBDA(const int64_t i, simd_t &lres)
                { lres += x(i) * y(i); }, agg);

            return horizontal_add<simd_t, double>(agg);

        }

        template <class TVector>
        static double norm2(const int N, const TVector &x) 
        {
            const size_t n = N / simd_t::size();

            // Accumulate per-lane and horizontal-add once.
            simd_t agg(0.0);
            Kokkos::parallel_reduce("NORM2", n, KOKKOS_LAMBDA(const int64_t i, simd_t &lres)
                { lres += x(i) * x(i); }, agg);

            return horizontal_add<simd_t, double>(agg);
        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            const size_t n = N / simd_t::size();
            Kokkos::parallel_for("DAXPY", n, KOKKOS_LAMBDA(const int64_t i)
                { y(i) = alpha * x(i) + y(i); });
        }
    };

    void run_test(int niter, int c);

    }

}

#endif