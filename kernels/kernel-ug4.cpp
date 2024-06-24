////////////////////////////////////////////////
// Tests for UG4.
////////////////////////////////////////////////
// #undef USE_UG4
#ifdef USE_UG4

#include <cstdlib>

// Fixtures.
#include "../meta/tools.hpp"
#include "../meta/fixtures.hpp"
#include "../meta/tests.hpp"

#include "kernel-ug4.hpp"


// Some dummies.
namespace ug {

	LogAssistant& LogAssistant::instance() {}
}

void  ug_assert_failed() {}



template <typename TAllocator>
void run_single_test_ug4(int niter, int c, size_t block_size=1)
{
	Fixture<TAllocator> f(niter, NVECTOR, c);
	f.SetUp();
	UnitTest_BLAS_Level1<myug4::mvops>(f.test,f.niter, f.n, block_size);
	PerfTest_BLAS_Level1<myug4::mvops>(f.test,f.niter, f.n, block_size);
	f.TearDown();

	MatrixVectorFixture<TAllocator> fdata(NCELLS);
	fdata.SetUp();
	UnitTest_BLAS_Level2<myug4::mvops>(50*niter, fdata, block_size);
	PerfTest_BLAS_Level2<myug4::mvops>(50*niter, fdata, block_size);
	fdata.TearDown();

}

#endif

//! Run all tests for UG4.
void run_test_ug4(int niter, int c)
{

#ifdef USE_UG4
	std::cout << "*** UG4-CPU1" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUAlgebra> TAllocator1;
	run_single_test_ug4<TAllocator1> (niter, c, 1);

	/*
	std::cout << "*** UG4-CPU2" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<2>> TAllocator2;
	run_single_test_ug4<TAllocator2> (niter, c, 2);

	std::cout << "*** UG4-CPU3" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<3>> TAllocator3;
	run_single_test_ug4<TAllocator3> (niter, c, 3);

	std::cout << "*** UG4-CPU4" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<4>> TAllocator4;
	run_single_test_ug4<TAllocator4> (niter, c, 4);

	std::cout << "*** UG4-CPU8" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<8>> TAllocator8;
	run_single_test_ug4<TAllocator8> (niter, c, 8);*/

/*
	std::cout << "UG4-CPU10" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<10>> TAllocator10;
	run_single_test_ug4<TAllocator10> (niter, c, 10);

	std::cout << "UG4-CPU16" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<16>> TAllocator16;
	run_single_test_ug4<TAllocator16> (niter, c, 16);

*/
#endif
}
