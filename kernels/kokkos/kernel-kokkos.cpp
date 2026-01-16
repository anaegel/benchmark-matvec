

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
	template <typename SizeType=size_t, typename Ordinal=size_t, typename Scalar=double>
	static void create_crs_matrix(int N, // number of cells 
		Kokkos::View<SizeType*> row_map, Kokkos::View<Ordinal*> columns, Kokkos::View<Scalar*> data)
	{
		// size:
		const SizeType numRows = N*N;

		row_map = Kokkos::View<SizeType*> ("row_map", numRows+1);
		columns = Kokkos::View<Ordinal*>  ("columns", numRows*5);
		data = Kokkos::View<Scalar*>  ("columns", numRows*5);
/*
		// Fill:
		for (size_t j=0; j<N; ++j)
		{
			// First "row", i=0
			const size_t ind  = ijindex(N, 0, j);
			tripletList.push_back(T(ind,ind, 1.0));
		}

		for (size_t i=1; i<N-1; ++i)
		{
			{
				const size_t ind  = ijindex(N, i, 0);
				tripletList.push_back(T(ind,ind, 1.0));
			}

			for (size_t j=1; j<N-1; ++j)
			{
				const size_t ind = ijindex(N, i, j);
				tripletList.push_back(T(ind,ind-N, -1.0));
				tripletList.push_back(T(ind,ind-1, -1.0));
				tripletList.push_back(T(ind,ind,	4.0));
				tripletList.push_back(T(ind,ind+1, -1.0));
				tripletList.push_back(T(ind,ind+N, -1.0));
			}

			{
				const size_t ind  = ijindex(N, i, N-1);
				tripletList.push_back(T(ind,ind, 1.0));
			}

		}

		for (size_t j=0; j<N; ++j)
		{
			const size_t ind = ijindex(N, N-1, j);
			tripletList.push_back(T(ind,ind, 1.0));  // set_matrix_row
		}

		*/
	}


	namespace plain
	{
		void run_test(int niter, int c)
		{
#ifdef USE_KOKKOS
			{
				using TVectorAllocator = mykokkos::plain::VectorAllocator;;
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
