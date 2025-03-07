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



// Fixtures (meta-programming)
#include "meta/fixtures.hpp"
#include "meta/tools.hpp"

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

// Kokkos.
#ifdef USE_MKL_BLAS
#include "kernels/kokkos/kernel-kokkos.hpp"
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


// Torch. 
#ifdef USE_KOKKOS
#include "kernels/kokkos/kernel-kokkos.hpp"
#endif



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

#ifdef USE_KOKKOS
    {
        Kokkos::initialize(argc, argv);
        std::cout << "*** Kokkos-Plain: " << std::endl;
        mykokkos::plain::run_test(niter, c);
        std::cout << "*** Kokkos-SIMD: " << std::endl;
        mykokkos::simd::run_test(niter, c);
    
    }

#endif

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
