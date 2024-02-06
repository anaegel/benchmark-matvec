

////////////////////////////////////////////////
// Tests for Eigen3.
////////////////////////////////////////////////

#ifdef USE_EIGEN3

// Fixtures.
#include "../meta/tools.hpp"
#include "../meta/fixtures.hpp"
#include "../meta/tests.hpp"

#include "kernel-eigen.hpp"

#endif

void run_test_eigen3(int niter, int c)
{
#ifdef USE_EIGEN3

		typedef Eigen::VectorXd TVector;

		// Vector tests
		Fixture<EigenVectorAllocator, TVector> f(niter, NVECTOR, c);
		f.SetUp();
		UnitTest_BLAS_Level1<myeigen::mvops>(f.test,f.niter, f.n);
		PerfTest_BLAS_Level1<myeigen::mvops>(f.test,f.niter, f.n);
		f.TearDown();

		// Matrix-vector tests
    	MatrixVectorFixture<EigenMatrixVectorAllocator> fdata(NCELLS);
    	fdata.SetUp();
    	UnitTest_BLAS_Level2<myeigen::mvops>(50*niter, fdata);
    	PerfTest_BLAS_Level2<myeigen::mvops>(50*niter, fdata);
    	fdata.TearDown();

#endif
}

