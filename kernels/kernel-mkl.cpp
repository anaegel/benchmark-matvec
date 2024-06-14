#ifdef USE_MKL_BLAS

#include "kernel-mkl.hpp"

const MKL_INT mymkl::mvops::one=1;


////////////////////////////////////////////////
// Tests for MKL.
////////////////////////////////////////////////

#endif

void run_test_mkl(int niter, int c)
{
    #ifdef USE_MKL_BLAS
	std::cout << "*** MKL-BLAS" << std::endl;

	typedef double* mkl_vector;
    Fixture<MKLMemoryAllocator> f(niter, NVECTOR, c);

    f.SetUp();
    UnitTest_BLAS_Level1<mymkl::mvops>(f.test,f.niter, f.n);
    PerfTest_BLAS_Level1<mymkl::mvops>(f.test,f.niter, f.n);
    f.TearDown();
    #endif

}



#endif
