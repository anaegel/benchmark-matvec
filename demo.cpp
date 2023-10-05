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


#undef GOOGLE_BENCHMARK
#ifdef GOOGLE_BENCHMARK
// Google benchmark
#include <benchmark/benchmark.h>
#endif

const int myAllocSize=64;

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
#define USE_CBLAS
#ifdef USE_CBLAS
#include "kernel-cblas.hpp"
#endif

// Intel MKL.
#undef USE_MKL_BLAS
#ifdef USE_MKL_BLAS
#include "kernel-mkl.hpp"
#endif


// Timing operations.
#include "timer.hpp"


#define NVECTOR 40000000


// This is a performance test for all functions.
template <typename F, typename V>
void PerformanceTest(V vec, size_t niter, size_t n)
{
   
    /* 1. dot */
    double s=0.0;
    TIMERSTART(tdot)
    for (size_t i=0; i<niter; ++i) {
        for (size_t j=0; j<niter; ++j) {
            s += F::dot(n, vec[i], vec[j]);
        }
    }
    TIMERSTOP(tdot, niter*niter*NVECTOR, 2*sizeof(double)*niter*niter*NVECTOR);
    std::cout << " for dot: " << s << " (for " << niter*niter <<" repetitions)" <<std::endl;



    /* 2. norm2 */
    const size_t nrep = 10;
    s=0.0;
    TIMERSTART(tnorm)
    for (size_t i=0; i<nrep*niter; ++i) {
        s += F::norm2(n, vec[i%niter]);
    }
    TIMERSTOP(tnorm, nrep*niter*NVECTOR, 1*sizeof(double)*nrep*niter*NVECTOR)
    std::cout << " for norm: " << s << std::endl;
    


    /* 3. daxpy */
    TIMERSTART(taxpy);
    for (size_t i=0; i<nrep*niter; ++i)
    {
        F::axpy(n, 2.0, vec[i%niter], vec[(i+1)%niter]);
    }
    TIMERSTOP(taxpy,nrep*niter*NVECTOR, 3*sizeof(double)*nrep*niter*NVECTOR)
    std::cout << " for axpy " << std::endl;

}


//!
struct StdArrayAllocator
{
	typedef double* TVector;

	static void allocate_vector (size_t n, TVector &p)
	{ p = (double*) aligned_alloc(myAllocSize, n*sizeof(double));}
	// { p = (double*) new(myAllocSize, n*sizeof(double));}

	static void  deallocate_vector(TVector &p)
	{ delete p;}
};



struct StdVectorAllocator
{
	typedef std::vector<double> TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.resize(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};



template <typename TVector>
void SetRandom(size_t n, int c, TVector &x)
{
	 for (size_t i=0; i<n; ++i)
	 { x[i] = 1.0*i*c; }
}




template <typename TAllocator, typename TVector = typename TAllocator::TVector>
struct Fixture {
    
    Fixture(int niter, size_t n, int c) : niter(niter), n(n), c(c) {}
    
    void SetUp()
    {
        test = new TVector[n+1];
        
        for (int i=0; i<=niter; ++i)
        {
        	TAllocator::allocate_vector(n, test[i]);
            SetRandom(n, c, test[i]);
        }
        
    }
    
    void TearDown()
    {
        for (int i=0; i<=niter; ++i)
        {
        	TAllocator::deallocate_vector(test[i]);
        }
      
        delete[] test;
    }
    
    ~Fixture() {}
    

    const int niter;		// Number of tests
    const size_t n;			// Size of test vector
    const int c;
    TVector* test;  		// Array of test vectors

};



template <typename TAllocator, typename TVector=typename TAllocator::TVector>
void run_test(int niter, int c)
{
    Fixture<TAllocator, TVector> f(niter, NVECTOR, c);
    f.SetUp();
    
    std::cout << "Manual" << std::endl;
    PerformanceTest<classic::mvops>(f.test, f.niter, f.n);
        
    std::cout << "SIMD" << std::endl;
    PerformanceTest<simd::mvops>(f.test,f.niter, f.n);
        
    std::cout << "OMP + SIMD" << std::endl;
    PerformanceTest<omp::mvops>(f.test,f.niter, f.n);
        
    std::cout << "BLAS" << std::endl;
    PerformanceTest<mycblas::mvops>(f.test, f.niter, f.n);
    
#ifdef USE_MKL_BLAS
    std::cout << "MKL" << std::endl;
    PerformanceTest<mymkl::mvops>(f.test,f.niter, f.n);
#endif
    
    f.TearDown();

}

#ifdef USE_EIGEN3
void run_test_eigen(int niter, int c)
{
    typedef Eigen::VectorXd TVector;
    Fixture<EigenVectorAllocator, TVector> f(niter, NVECTOR, c);
    f.SetUp();

    std::cout << "Eigen" << std::endl;
    PerformanceTest<myeigen::mvops>(f.test,f.niter, f.n);

    f.TearDown();


    {
    // 4. maxpy
    // a) Fixtures
    const size_t npoints = 2000;
    const size_t nvector = npoints*npoints;
    TVector x(nvector);
    TVector b(nvector);
    typedef EigenVectorAllocator::TMatrix TMatrix;
    TMatrix* mat = EigenVectorAllocator::create_matrix(2000);
    EigenVectorAllocator::create_matrix(npoints);



    // b) Test
    size_t nrep = 100*niter;

    const size_t NSTENCIL = 5;


    {

    	 // y = A*x
    	 const size_t mem_cnt = nrep*(NSTENCIL+1)*nvector*sizeof(double);  //
    	 const size_t flop_cnt = nrep*(NSTENCIL+1)*nvector;

    	TIMERSTART(tmatmul);
    	for (size_t i=0; i<nrep; ++i)
    	{
    		myeigen::mvops::matmul_set(nvector, b, *mat, x);
    	}
    	TIMERSTOP(tmatmul,  flop_cnt, mem_cnt)
    	std::cout << " for matmul (" << nrep <<" products)"<< std::endl;
    }

    {

    	// y = y + A*x
    	 const size_t mem_cnt = nrep*(NSTENCIL+1)*nvector*sizeof(double);  //
    	 const size_t flop_cnt = nrep*(NSTENCIL+1)*nvector;

       	TIMERSTART(tmatmul);
       	for (size_t i=0; i<nrep; ++i)
       	{
       		myeigen::mvops::matmul_add(nvector, b, *mat, x);
       	}
       	TIMERSTOP(tmatmul, flop_cnt, mem_cnt)
       	std::cout << " for matmul (" << nrep <<" products)"<< std::endl;
     }
    }
}
#endif



void run_test_ug4(int niter, int c)
{

	{
		typedef UG4Allocator::TVector TVector;
		Fixture<UG4Allocator, TVector> f(niter, NVECTOR, c);
		f.SetUp();

		std::cout << "UG4-CPU1" << std::endl;
		PerformanceTest<myug4::mvops>(f.test,f.niter, f.n);

		f.TearDown();
	}

	{
		typedef UG4AllocatorBlock<2> TAlgebra3;
		Fixture<TAlgebra3> f(niter, NVECTOR, c);
		f.SetUp();

		std::cout << "UG4-CPU2" << std::endl;
		PerformanceTest<myug4::mvops>(f.test,f.niter, f.n);

		f.TearDown();
	}

/*	{
		typedef UG4AllocatorBlock<3> TAlgebra3;
		Fixture<TAlgebra3> f(niter, NVECTOR, c);
		f.SetUp();

		std::cout << "UG4-CPU3" << std::endl;
		PerformanceTest<myug4::mvops>(f.test,f.niter, f.n);
		f.TearDown();
	}
*/
	{
		typedef UG4AllocatorBlock<4> TAlgebra4;
		Fixture<TAlgebra4> f(niter, NVECTOR, c);
		f.SetUp();

		std::cout << "UG4-CPU4" << std::endl;
		PerformanceTest<myug4::mvops>(f.test,f.niter, f.n);

		f.TearDown();
	}


}


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


#ifdef USE_EIGEN3
    {
        std::cout << "For eigen: " << std::endl;
        run_test_eigen(niter, c);
    }
#endif
    

#ifdef USE_UG4
    {
           std::cout << "UG4: " << std::endl;
           run_test_ug4(niter, c);
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


#ifdef USE_MPI
    MPI_Finalize();
#endif
}
    
   
#endif
