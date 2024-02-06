// Testsuite zur Optimierung von Matrix-Vektorroutinen.
// (c) G-CSC, Uni Frankfurt, Winter 2021/22.

// Stdlib.
#include <chrono>
#include <iostream>
#include <time.h>       /* time */
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include <vector>
#include <cstdlib>



#include "meta/tests.hpp"



#undef GOOGLE_BENCHMARK
#ifdef GOOGLE_BENCHMARK
// Google benchmark
#include <benchmark/benchmark.h>
#endif

const int myAllocSize=64;

// Fixtures (meta-programming)
#include "meta/fixtures.hpp"

// Default kernel.
#include "kernels/kernel-default.hpp"

// Eigen3.
#ifdef USE_EIGEN3
#include "kernels/kernel-eigen.hpp"
#endif

// UG4.
#ifdef USE_UG4
#include "kernels/kernel-ug4.hpp"
#endif

// CBLAS.
// #undef USE_CBLAS
#ifdef USE_CBLAS
#include "kernels/kernel-cblas.hpp"
#endif

// BLAS.
#ifdef USE_BLAS
#include "kernels/kernel-blas.hpp"
#endif


// Intel MKL.
#ifdef USE_MKL_BLAS
#include "kernels/kernel-mkl.hpp"
// const int mymkl::mvops::one = 1;
#endif


// SYCL.
#ifdef USE_SYCL
#include "kernels/kernel-sycl.hpp"
#endif

// OpenCL.
#undef USE_OPENCL
#ifdef USE_OPENCL
#include "kernels/opencl/kernel-opencl.hpp"
#endif

// Apple Metal.
#ifdef USE_METAL
#include "kernels/metal/kernel-metal.hpp"
#endif


// Torch. 
#ifdef USE_TORCH
#include "kernels/torch/kernel-torch.hpp"
#endif


//!
struct StdArrayAllocator
{
	typedef double* TVector;

	static void allocate_vector (size_t n, TVector &p)
	{ 

        #ifdef _WIN32
            p = (double*) _aligned_malloc(n*sizeof(double), myAllocSize);
        #else 
           p = (double*) aligned_alloc(myAllocSize, n*sizeof(double));
           // p = new double[n];
        #endif
        //p = (double*) _mm_alloc( n * sizeof(double), myAllocSize);
    }


	static void  deallocate_vector(TVector &p)
	{ 
         #ifdef _WIN32
             _aligned_free(p);
        #else 
            free(p);
            // delete p;
        #endif
       
        //_mm_free(p);
    }
};



struct StdVectorAllocator
{
	typedef std::vector<double> TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.resize(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};




////////////////////////////////////////////////
// General test suite.
////////////////////////////////////////////////

template <typename TAllocator, typename TVector=typename TAllocator::TVector>
void run_test(int niter, int c)
{
    Fixture<TAllocator, TVector> f(niter, NVECTOR, c);
    f.SetUp();    

    std::cout << "*** Manual" << std::endl;
    UnitTest_BLAS_Level1<classic::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<classic::mvops>(f.test, f.niter, f.n);

 #ifdef USE_OPENMP
    std::cout << "*** SIMD" << std::endl;
    UnitTest_BLAS_Level1<simd::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<simd::mvops>(f.test, f.niter, f.n);
        
    std::cout << "*** OMP + SIMD" << std::endl;
    UnitTest_BLAS_Level1<omp::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<omp::mvops>(f.test, f.niter, f.n);
#endif

#ifdef USE_BLAS
    std::cout << "*** BLAS" << std::endl;
    PerfTest_BLAS_Level1<myblas::mvops>(f.test,f.niter, f.n);
#endif

#ifdef USE_CBLAS
    std::cout << "*** CBLAS" << std::endl;
    UnitTest_BLAS_Level1<mycblas::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<mycblas::mvops>(f.test, f.niter, f.n);
#endif

    f.TearDown();

}

////////////////////////////////////////////////
// Tests for MKL.
////////////////////////////////////////////////

#ifdef USE_MKL_BLAS
void run_test_mkl(int niter, int c)
{
	std::cout << "*** MKL-BLAS" << std::endl;

	typedef double* mkl_vector;
    Fixture<MKLMemoryAllocator> f(niter, NVECTOR, c);

    f.SetUp();
    UnitTest_BLAS_Level1<mymkl::mvops>(f.test,f.niter, f.n);
    PerfTest_BLAS_Level1<mymkl::mvops>(f.test,f.niter, f.n);
    f.TearDown();

}
#endif



////////////////////////////////////////////////
// Tests for Eigen3.
////////////////////////////////////////////////

#ifdef USE_EIGEN3
void run_test_eigen3(int niter, int c)
{
	{

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
	}

}
#endif


////////////////////////////////////////////////
// Tests for UG4.
////////////////////////////////////////////////
#undef USE_UG4
#ifdef USE_UG4

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


//! Run all tests for UG4.
void run_test_ug4(int niter, int c)
{

	std::cout << "*** UG4-CPU1" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUAlgebra> TAllocator1;
	run_single_test_ug4<TAllocator1> (niter, c, 1);

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
	run_single_test_ug4<TAllocator8> (niter, c, 8);

/*
	std::cout << "UG4-CPU10" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<10>> TAllocator10;
	run_single_test_ug4<TAllocator10> (niter, c, 10);

	std::cout << "UG4-CPU16" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<16>> TAllocator16;
	run_single_test_ug4<TAllocator16> (niter, c, 16);

*/
}
#endif


#ifdef GOOGLE_BENCHMARK


template <typename TFixture>
class BenchmarkFixture : public TFixture, ::benchmark::Fixture
{

public:
    void SetUp(const ::benchmark::State& st)
    {}

    void TearDown(const ::benchmark::State&)
    {}

};


// Define another benchmark
static void BM_Dot(benchmark::State& state) {
}

BENCHMARK(BM_Dot);

BENCHMARK_MAIN();


#else


// This is a custom main.
int main(int argc, char* argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc,&argv);
#endif
    
    
    char *myarg= argv[1];
    const int niter = atoi(myarg);
    std::cout << niter << std::endl;
    
    std::srand(time(NULL));
    int c =atoi(argv[2]);

#ifdef USE_OPENMP
    std::cout << "OMP_NUM_THREADS: " << omp_get_num_threads() << std::endl;
    std::cout << "OMP_NUM_PROCS: " << omp_get_num_procs() << std::endl;
    std::cout << "OMP_MAX_THREADS: " << omp_get_max_threads() << std::endl;
#endif



#ifdef USE_SYCL
    {
        std::cout << "*** SYCL: " << std::endl;
        run_test_sycl(niter, c);
    }
#endif

    {
        std::cout << "For double* " << std::endl;
        run_test<StdArrayAllocator> (niter, c);
    }

    {
        std::cout << "For std::vector<double>: " << std::endl;
        run_test<StdVectorAllocator> (niter, c);
    }

#ifdef USE_EIGEN3
    {
        std::cout << "*** Eigen3: " << std::endl;
        run_test_eigen3(niter, c);
    }
#endif

#ifdef USE_TORCH
    {
        std::cout << "*** TORCH: " << std::endl;
        run_test_torch(niter, c, NVECTOR);
    }
#endif

#ifdef USE_UG4
    {
           std::cout << "UG4: " << std::endl;
           run_test_ug4(niter, c);
    }
#endif




#ifdef USE_MPI
    MPI_Finalize();
#endif
}
    
   
#endif
