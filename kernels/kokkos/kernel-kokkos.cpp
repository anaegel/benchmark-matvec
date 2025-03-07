

////////////////////////////////////////////////
// Tests for Kokkos
////////////////////////////////////////////////

#ifdef USE_KOKKOS

// Fixtures.
#include "../../meta/tools.hpp"
#include "../../meta/fixtures.hpp"
#include "../../meta/tests.hpp"

#include "kernel-kokkos-simd.hpp"
#include "kernel-kokkos-plain.hpp"
#endif

namespace mykokkos
{


	namespace plain
	{
		void run_test(int niter, int c)
		{
#ifdef USE_KOKKOS
			//
			{
				using TVectorAllocator = StdArrayAllocator;
				using TVector = TVectorAllocator::TVector;
				// typedef KokkosVectorAllocator TVectorAllocator;

				// Vector tests
				Fixture<TVectorAllocator, TVector> f(niter, NVECTOR, c);

				f.SetUp();
				UnitTest_BLAS_Level1<mvops>(f.test, f.niter, f.n);
				PerfTest_BLAS_Level1<mvops>(f.test, f.niter, f.n);
				f.TearDown();

				// Matrix-vector tests
				/*MatrixVectorFixture<StdArrayAllocator> fdata(NCELLS);
				fdata.SetUp();
				UnitTest_BLAS_Level2<mykokkos::mvops>(50*niter, fdata);
				PerfTest_BLAS_Level2<mykokkos::mvops>(50*niter, fdata);
				fdata.TearDown();*/
			}
#endif
		}

	}

	namespace simd
	{
		void run_test(int niter, int c)
		{
#ifdef USE_KOKKOS
			//
			{
				using TVectorAllocator = mykokkos::simd::VectorAllocator;
				using TVector = TVectorAllocator::TVector;
				// typedef KokkosVectorAllocator TVectorAllocator;

				// Vector tests
				Fixture<TVectorAllocator, TVector> f(niter, NVECTOR, c);

				f.SetUp();
				UnitTest_BLAS_Level1<mvops>(f.test, f.niter, f.n);
				PerfTest_BLAS_Level1<mvops>(f.test, f.niter, f.n);
				f.TearDown();

				// Matrix-vector tests
				/*MatrixVectorFixture<StdArrayAllocator> fdata(NCELLS);
				fdata.SetUp();
				UnitTest_BLAS_Level2<mykokkos::mvops>(50*niter, fdata);
				PerfTest_BLAS_Level2<mykokkos::mvops>(50*niter, fdata);
				fdata.TearDown();*/
			}
#endif
		}

	}

}
