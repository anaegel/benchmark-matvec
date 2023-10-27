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


#undef GOOGLE_BENCHMARK
#ifdef GOOGLE_BENCHMARK
// Google benchmark
#include <benchmark/benchmark.h>
#endif

const int myAllocSize=64;

// Fixtures (meta-programming)
#include "fixture-metas.hpp"

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
#undef USE_CBLAS
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
#undef USE_SYCL
#ifdef USE_SYCL
#include "kernels/kernel-sycl.hpp"
#endif


// Timing operations.
#include "timer.hpp"

#define NCELLS 2000
#define NVECTOR 40000000

////////////////////////////////////////////////
// BLAS - Level 2
////////////////////////////////////////////////

// Unit tests.
template <typename F, typename V>
void UnitTest_BLAS_Level1(V vec, size_t niter, size_t n, size_t block_size=1)
{
	{

		for (size_t i=0; i<n; ++i) { vec[0][i] = 1.0; vec[1][i] = 0.0;}
		std::cout << "Level 1 - UnitTest: "
				  <<  F::dot(n, vec[0], vec[1]) << ", "
				  <<  F::norm2(n, vec[0]) << ", ";

		F::axpy(n, 2.0, vec[0], vec[1]);

		std::cout <<  F::norm2(n, vec[1]) << std::endl;

	}
}

// This is a performance test for all functions.
template <typename F, typename V>
void PerfTest_BLAS_Level1(V vec, size_t niter, size_t n, size_t block_size=1)
{
   const size_t nentries = NVECTOR*block_size;
    /* 1. dot */
    double s=0.0;
    TIMERSTART(tdot)
    for (size_t i=0; i<niter; ++i) {
        for (size_t j=0; j<niter; ++j) {
            s += F::dot(n, vec[i], vec[j]);
        }
    }
    TIMERSTOP(tdot, niter*niter*nentries, 2*sizeof(double)*niter*niter*nentries);
    std::cout << " for dot: " << s << " (for " << niter*niter <<" repetitions)" <<std::endl;



    /* 2. norm2 */
    const size_t nrep = 10;
    s=0.0;
    TIMERSTART(tnorm)
    for (size_t i=0; i<nrep*niter; ++i) {
        s += F::norm2(n, vec[i%niter]);
    }
    TIMERSTOP(tnorm, nrep*niter*nentries, 1*sizeof(double)*nrep*niter*nentries)
    std::cout << " for norm: " << s << std::endl;
    


    /* 3. daxpy */
    TIMERSTART(taxpy);
    for (size_t i=0; i<nrep*niter; ++i)
    {
        F::axpy(n, 2.0, vec[i%niter], vec[(i+1)%niter]);
    }
    TIMERSTOP(taxpy,nrep*niter*nentries, 3*sizeof(double)*nrep*niter*nentries)
    std::cout << " for axpy " << std::endl;

}

////////////////////////////////////////////////
// BLAS - Level 2
////////////////////////////////////////////////

// This is a performance test for all functions.
template <typename TFunctions, typename TFixture>
void UnitTest_BLAS_Level2(const size_t nrep, TFixture &f, size_t block_size=1)
{

	std::cout << "Level 2 - Unit Test: ";
	{
		for (size_t i=0; i<f.n; ++i) { f.x[i] = 1.0; }
		TFunctions::matmul(f.n, f.b, *(f.A), f.x);
		double val = TFunctions::norm2(f.n, f.b);
		std::cout << val/f.n << ", ";
	}

	{
		for (size_t i=0; i<f.n; ++i) { f.x[i] = 1.0; }
		TFunctions::matmul_transposed(f.n, f.b, *(f.A), f.x);
		double val = TFunctions::norm2(f.n, f.b);
		std::cout << val/f.n;
	}
	std::cout <<  std::endl;

}


// This is a performance test for all functions.
template <typename TFunctions, typename TFixture>
void PerfTest_BLAS_Level2(const size_t nrep, TFixture &f, size_t block_size=1)
{
	const size_t NSTENCIL = 5;
	const size_t nvector = f.n;

	const size_t m2 = block_size*block_size;

	// Each matrix entry is multiplied by a vector entry the vector. The product is added to the dest.
	const size_t flop_cnt_per_row = ((m2*NSTENCIL) + block_size* NSTENCIL) ;

	// For each block row, we must collect (m^2 * NSTENCIL) matrix entries and 2*m vector entries
	const size_t mem_cnt_per_row = sizeof(double) *   // for each block_row:
			( m2*NSTENCIL + 2* block_size);  //

	{	// 4. y = A*x

		TIMERSTART(tmatmul);
		for (size_t i=0; i<nrep; ++i)
		{ TFunctions::matmul(f.n, f.b, *(f.A), f.x); }

		TIMERSTOP(tmatmul, nrep*flop_cnt_per_row*nvector, nrep*mem_cnt_per_row*nvector);
		std::cout << " for matmul (" << nrep <<" products)"<< std::endl;
	}

	{	// 4. y = A^T*x

			TIMERSTART(tmatmul);
			for (size_t i=0; i<nrep; ++i)
			{ TFunctions::matmul_transposed(f.n, f.b, *(f.A), f.x); }

			TIMERSTOP(tmatmul, nrep*flop_cnt_per_row*nvector, nrep*mem_cnt_per_row*nvector);
			std::cout << " for matmul_transposed (" << nrep <<" products)"<< std::endl;
	}

	{	// 5. y += A*x
		TIMERSTART(tmatmul);
		for (size_t i=0; i<nrep; ++i)
		{ TFunctions::matmul_add(f.n, f.b, *(f.A), f.x); }

		TIMERSTOP(tmatmul, nrep*flop_cnt_per_row*nvector, nrep*mem_cnt_per_row*nvector)
		std::cout << " for matmul (" << nrep <<" products)"<< std::endl;
	}



}


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

    std::cout << "Manual" << std::endl;
    UnitTest_BLAS_Level1<classic::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<classic::mvops>(f.test, f.niter, f.n);

 #ifdef USE_OPENMP
    std::cout << "SIMD" << std::endl;
    UnitTest_BLAS_Level1<simd::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<simd::mvops>(f.test, f.niter, f.n);
        
    std::cout << "OMP + SIMD" << std::endl;
    UnitTest_BLAS_Level1<omp::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<omp::mvops>(f.test, f.niter, f.n);
#endif

#ifdef USE_BLAS
    std::cout << "BLAS" << std::endl;
    PerfTest_BLAS_Level1<myblas::mvops>(f.test,f.niter, f.n);
#endif

#ifdef USE_CBLAS
    std::cout << "CBLAS" << std::endl;
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
	std::cout << "MKL-BLAS" << std::endl;

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

	std::cout << "UG4-CPU1" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUAlgebra> TAllocator1;
	run_single_test_ug4<TAllocator1> (niter, c, 1);

	std::cout << "UG4-CPU2" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<2>> TAllocator2;
	run_single_test_ug4<TAllocator2> (niter, c, 2);

	std::cout << "UG4-CPU3" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<3>> TAllocator3;
	run_single_test_ug4<TAllocator3> (niter, c, 3);

	std::cout << "UG4-CPU4" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<4>> TAllocator4;
	run_single_test_ug4<TAllocator4> (niter, c, 4);

	std::cout << "UG4-CPU8" << std::endl;
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
    
    // std::cout << omp_get_num_procs() << std::endl;
    // std::cout << omp_get_num_threads() << std::endl;
    
    char *myarg= argv[1];
    const int niter = atoi(myarg);
    std::cout << niter << std::endl;
    

    std::srand(time(NULL));
    int c =atoi(argv[2]);

    std::cout << "OMP_NUM_THREADS: " << omp_get_num_threads() << std::endl;
    std::cout << "OMP_NUM_PROCS: " << omp_get_num_procs() << std::endl;
    std::cout << "OMP_MAX_THREADS: " << omp_get_max_threads() << std::endl;

#ifdef USE_SYCL
    {
        std::cout << "SYCL: " << std::endl;
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
        std::cout << "Eigen3: " << std::endl;
        run_test_eigen3(niter, c);
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
