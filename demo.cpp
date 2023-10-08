// Testsuite zur Optimierung von Matrix-Vektorroutinen.
// (c) G-CSC, Uni Frankfurt, Winter 2021/22.

// Stdlib.
#include <omp.h>
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
#include "kernel-default.hpp"

// Eigen3.
#ifdef USE_EIGEN3
#include "kernel-eigen.hpp"
#endif

// UG4.
#ifdef USE_UG4
#include "kernel-ug4.hpp"
#endif

// CBLAS.
#undef USE_CBLAS
#ifdef USE_CBLAS
#include "kernel-cblas.hpp"
#endif

// Intel MKL.
#ifdef USE_MKL_BLAS
#include "kernel-mkl.hpp"
#endif


// Timing operations.
#include "timer.hpp"

#define NCELLS 2000
#define NVECTOR 40000000


// This is a performance test for all functions.
template <typename F, typename V>
void PerformanceTestVector(V vec, size_t niter, size_t n, size_t block_size=1)
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

// Intel SYCL.
#ifdef USE_SYCL
#include "kernel-sycl.hpp"
#endif


// This is a performance test for all functions.
template <typename TFunctions, typename TFixture>
void PerformanceMV(const size_t nrep, TFixture &f, size_t block_size=1)
{
	const size_t NSTENCIL = 5;
	const size_t nvector = f.n;

	const size_t m2 = block_size*block_size;

	// Each matrix entry is multiplied by a vector entry the vector. The product is added to the dest.
	const size_t flop_cnt_per_row = ((m2*NSTENCIL) + block_size* NSTENCIL) ;

	// For each block row, we must collect (m^2 * NSTENCIL) matrix entries and 2*m vector entries
	const size_t mem_cnt_per_row = sizeof(double) *   // for each block_row:
			( block_size*block_size*NSTENCIL + 2* block_size);  //

	{	// 4. y = A*x
		const size_t flop_cnt = (NSTENCIL+1)*nvector;
		const size_t mem_cnt = (NSTENCIL+1)*nvector*sizeof(double);  // each m^2

		TIMERSTART(tmatmul);
		for (size_t i=0; i<nrep; ++i)
		{ TFunctions::matmul_set(f.n, f.b, *(f.A), f.x); }

		TIMERSTOP(tmatmul, nrep*flop_cnt_per_row*nvector, nrep*mem_cnt_per_row*nvector);
		std::cout << " for matmul (" << nrep <<" products)"<< std::endl;
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
        //p = (double*) aligned_alloc(myAllocSize, n*sizeof(double));
        p = (double*) _aligned_malloc(n*sizeof(double), myAllocSize);
        //p = (double*) _mm_alloc( n * sizeof(double), myAllocSize);
    }
	// { p = (double*) new(myAllocSize, n*sizeof(double));}

	static void  deallocate_vector(TVector &p)
	{ 
        // delete p;
        _aligned_free(p);
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






template <typename TAllocator, typename TVector=typename TAllocator::TVector>
void run_test(int niter, int c)
{
    Fixture<TAllocator, TVector> f(niter, NVECTOR, c);
    f.SetUp();
    
    PerformanceTestVector<mysycl::mvops>(f.test, f.niter, f.n);

    std::cout << "Manual" << std::endl;
    PerformanceTestVector<classic::mvops>(f.test, f.niter, f.n);
        
    std::cout << "SIMD" << std::endl;
    PerformanceTestVector<simd::mvops>(f.test,f.niter, f.n);
        
    std::cout << "OMP + SIMD" << std::endl;
    PerformanceTestVector<omp::mvops>(f.test,f.niter, f.n);
#ifdef USE_CBLAS     
    std::cout << "BLAS" << std::endl;
    PerformanceTestVector<mycblas::mvops>(f.test, f.niter, f.n);
#endif
#ifdef USE_MKL_BLAS
    std::cout << "MKL" << std::endl;
    PerformanceTestVector<mymkl::mvops>(f.test,f.niter, f.n);
#endif
    
    f.TearDown();

}

#ifdef USE_EIGEN3
void run_test_eigen3(int niter, int c)
{
	{

		typedef Eigen::VectorXd TVector;

		// Vector tests
		Fixture<EigenVectorAllocator, TVector> f(niter, NVECTOR, c);
		f.SetUp();
		PerformanceTestVector<myeigen::mvops>(f.test,f.niter, f.n);
		f.TearDown();

		// Matix-vector tests
    	MatrixVectorFixture<EigenMatrixVectorAllocator> fdata(NCELLS);
    	fdata.SetUp();
    	PerformanceMV<myeigen::mvops>(50*niter, fdata);
    	fdata.TearDown();
	}

}
#endif


#ifdef USE_UG4

template <typename TAllocator>
void run_single_test_ug4(int niter, int c, size_t block_size=1)
{
	Fixture<TAllocator> f(niter, NVECTOR, c);
	f.SetUp();
	PerformanceTestVector<myug4::mvops>(f.test,f.niter, f.n, block_size);
	f.TearDown();

	MatrixVectorFixture<TAllocator> fdata(NCELLS);
	fdata.SetUp();
	PerformanceMV<myug4::mvops>(50*niter, fdata, block_size);
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

    /*
	std::cout << "UG4-CPU4" << std::endl;
	typedef UG4AlgebraAllocator<ug::CPUBlockAlgebra<4>> TAllocator4;
	run_single_test_ug4<TAllocator4> (niter, c, 4);
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
