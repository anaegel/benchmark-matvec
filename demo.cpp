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

#define USE_EIGEN3
#ifdef USE_EIGEN3
#include <Eigen/Sparse>
#endif

#undef GOOGLE_BENCHMARK
#ifdef GOOGLE_BENCHMARK
// Google benchmark
#include <benchmark/benchmark.h>
#endif

const int myAllocSize=64;

#ifdef USE_MPI
#include <mpi>
#endif


// #include <Accelerate/Accelerate.h>
#include <vecLib/vecLib.h>
// #include <vecLib/cblas.h>

#include "timer.hpp"

#include "kernel-eigen.hpp"
// #include "kernel-mkl.hpp"



#define NVECTOR 40000000


// Klassische Routinen.
namespace classic {

    struct mvops {
    
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            for (int i=0; i<N; ++i) { sum += x[i]*y[i]; }
            return sum;
        }
        
        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }
        
        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            for (int i=0; i<N; ++i)  { y[i] = alpha*x[i] + y[i]; }
        }

    };
}


//! Nun mit SIMD.
namespace simd {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int i=0; i<N; ++i)
            { sum += x[i]*y[i];}
            return sum;
        }
        
        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }
        
        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            #pragma omp simd
            for (int i=0; i<N; ++i)
            { y[i] = alpha*x[i] + y[i];}
        }
    };

}


//! Nun mit OpenMP+SIMD
namespace omp {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
            double sum = 0.0;
            #pragma omp parallel for simd shared(x,y,N) schedule(static) reduction(+:sum)
            for (int i=0; i<N; ++i)
            { sum += x[i]*y[i]; }
            return sum;
        }
        
        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return dot(N,x,x); }
        
        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
            #pragma omp parallel for simd shared(x,y,N) schedule(static)
            for (int i=0; i<N; ++i)
            { y[i] = alpha*x[i] + y[i];}
        }
    };

}



#define USE_CBLAS
#ifdef USE_CBLAS
// CBLAS
namespace mycblas {

    struct mvops {
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        { return cblas_ddot(N, &x[0], 1, &y[0], 1); }
        
        template <class TVector>
        static double norm2(const int N, const TVector &x)
        { return cblas_dnrm2(N, &x[0], 1); }
        
        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        { cblas_daxpy(N, alpha, &x[0], 1, &y[0], 1); }
    };

}
#endif




#ifdef USE_MPI

namespace mpi {
struct mvops {
// MPI inner product
template <class TVector>
double dot_mpi(const int n, const TVector &x, const TVector &y, const MPI_Comm icomm)
{
  double s = dot_simd(n,x,y);
    // call sequential inner product
  double sg;
  MPI_Allreduce(&s,&sg,1,MPI_DOUBLE,MPI_SUM,icomm);
  return(sg);
}
#endif






template <typename F, typename V>
void PerformanceTest(V vec, size_t niter, size_t n)
{
   
    /* dot */
    double s=0.0;
    TIMERSTART(tdot)
    for (size_t i=0; i<niter; ++i) {
        for (size_t j=0; j<niter; ++j) {
            s += F::dot(n, vec[i], vec[j]);
        }
    }
    TIMERSTOP(tdot, niter*niter*NVECTOR, 2*sizeof(double)*niter*niter*NVECTOR);
    std::cout << " for dot: " << s << std::endl;

  
    /* norm2 */
    const size_t nrep = 10;
    s=0.0;
    TIMERSTART(tnorm)
    for (size_t i=0; i<nrep*niter; ++i) {
        s += F::norm2(n, vec[i%niter]);
    }
    TIMERSTOP(tnorm, nrep*niter*NVECTOR, 1*sizeof(double)*nrep*niter*NVECTOR)
    std::cout << " for norm: " << s << std::endl;
    
    /* daxpy */
    TIMERSTART(taxpy);
    for (size_t i=0; i<nrep*niter; ++i)
    {
        F::axpy(n, 2.0, vec[i%niter], vec[(i+1)%niter]);
    }
    TIMERSTOP(taxpy,nrep*niter*NVECTOR, 3*sizeof(double)*nrep*niter*NVECTOR)
    std::cout << " for axpy " << std::endl;

}

#undef GOOGLE_BENCHMARK
#ifdef GOOGLE_BENCHMARK



template <class Part>
class BenchmarkFixture : public ::benchmark::Fixture {
    
public:
    void SetUp(const ::benchmark::State& st)
    {
        c = rand();
        
        for (size_t i=0; i<=niter; ++i)
        {
            // Diese Funktion ist für beide Faelle ueberladen!
             InitArray(n, test[i]);
             SetRandomArray(n, c, test[i]);
        }
    }

    void TearDown(const ::benchmark::State&)
    {
        for (int i=0; i<=niter; ++i)
            delete test[i];
    }
    
    int c;
    const size_t niter = 20;
    double* test[niter+1];

    
   
}


// Define another benchmark
static void BM_Dot(benchmark::State& state) {
  
    const size_t n = NVECTOR;
    int c =atoi(argv[2]);

    // Erzeuge einige Vektoren.
    //std::vector<double> test[niter+1];        // Benutze std::vector
    double* test[niter+1];                    // Benutze classic arrays.
    for (size_t i=0; i<=niter; ++i){
         CreateRandomArray(n, c, test[i]);     // Diese Funktion ist für beide Faelle ueberladen!
    }
      std::cout << "Classic" << std::endl;
      PerformanceTest<classic::mvops>(test,niter, n);
    
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_Dot);

BENCHMARK_MAIN();
#else

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
    
    Fixture(int niter, size_t n) : niter(niter), n(n) {}
    
    void SetUp(int c)
    {
        test = new TVector[n+1];
        
        for (int i=0; i<=niter; ++i){
        	TAllocator::allocate_vector(n, test[i]);
            SetRandom(n, c, test[i]);
        }
        
    }
    


    void TearDown()
    {
        for (int i=0; i<=niter; ++i) {
        	TAllocator::deallocate_vector(test[i]);
        }
      
        delete[] test;
    }
    
    ~Fixture() {}
    
    const int niter;		// Number of tests
    const size_t n;			// Size of test vector
    TVector* test;  		// Array of test vectors



};



template <typename TAllocator, typename TVector=typename TAllocator::TVector>
void run_test(int niter, int c)
{
    Fixture<TAllocator, TVector> f(niter, NVECTOR);
    f.SetUp(c);
    
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
    
    
#ifdef USE_MPI
    MPI_Finalize();
#endif
    
    f.TearDown();

}

#ifdef USE_EIGEN3
void run_test_eigen(int niter, int c)
{
    typedef Eigen::VectorXd TVector;
    Fixture<EigenVectorAllocator, TVector> f(niter, NVECTOR);
    f.SetUp(c);
    
   
    std::cout << "Eigen" << std::endl;
    PerformanceTest<myeigen::mvops>(f.test,f.niter, f.n);
    f.TearDown();
    
}
#endif

// This is a custom main.
int main(int argc, char* argv[])
{
#ifdef USE_MPI
    MPI_Init(&argc,&argv);
#endif
    
    // std::cout << omp_get_num_procs() << std::endl;
    // std::cout << omp_get_num_threads() << std::endl;
    
    char *myarg= argv[1];
    int c =atoi(argv[2]);
    
    const int niter = atoi(myarg);
    std::cout << niter << std::endl;
    
    std::srand(time(NULL));

#ifdef USE_EIGEN3
    {
        std::cout << "For eigen: " << std::endl;
        run_test_eigen(niter, c);
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
}
    
   
#endif
