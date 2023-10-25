#pragma once

// Include SYCL.
#include <CL/sycl.hpp>
using namespace cl;

// Fixtures.
#include "fixture-metas.hpp"



// Aligned memory allocation for MKL.
struct SyclMemoryAllocator
{
	typedef double* TVector;

	static void allocate_vector   (size_t n, TVector &v)
	{ v = (double *) malloc(n *sizeof(double)); }

	static void deallocate_vector (TVector &v)
	{ free(v); }
 
   //  void set_queue(sycl::queue* Q) {q=Q;}
   // static sycl::queue q;
};


struct SyclFixture : public Fixture<SyclMemoryAllocator>
{
    typedef Fixture<SyclMemoryAllocator> fixture_type;
   
    SyclFixture(int niter, size_t n, int c)
    : fixture_type(niter, n, c) {}
    
    void SetUp()
    { fixture_type::SetUp(); }

    void TearDown()
    { fixture_type::TearDown(); }
    
    //
};


// MKL-BLAS interface.
namespace mysycl {

    struct mvops {
    	static const int one = 1;
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
           sycl::queue q;
            
            std::cout << "dot:" << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    
            sycl::buffer<double,1> xbuf(&x[0], N);
            sycl::buffer<double,1> ybuf(&y[0], N);
            
            // Initialize zero.
            double* dresult = sycl::malloc_shared<double>(1, q);
            dresult[0] = 0.0;
            /*sycl::buffer<double,1> bresult(dresult, 1);
            q.submit( [&](sycl::handler& h) {
                sycl::accessor res(bresult, h, sycl::write_only);
                h.single_task([=]() {res[0] = 0.0;});
            });
            q.wait();
*/
            // Compute inner product.
            /*q.submit([&] (auto &h) {
                auto ax = xbuf.get_access<sycl::access::mode::read> (h);
                auto ay = ybuf.get_access<sycl::access::mode::read> (h);
                         
                auto red = sycl::reduction(bresult, h, sycl::plus<>());
                // auto red = sycl::reduction(bresult, h, 0.0, sycl::plus<double>());
                h.parallel_for(sycl::range<1>(N), red, [=](sycl::id<1> i, auto &sum) {
                    sum += ax[i] * ay[i];;
                });
            });
*/
            // Transfer result to host and synchronize
            // Note: host_accessor is blocking (thus can be used to synchronize)
           /* sycl::host_accessor res(bresult, sycl::read_only);
            return dresult[0];
            */
           return 0.0;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        {
            return 0.0;
            // return dot<TVector>(N,x,x);
        }

        template <class TVector>
        static void axpy(const int N, double alpha, const TVector &x, TVector &y)
        {
          sycl::queue q;

          sycl::buffer<double,1> xbuf(&x[0], N);
          sycl::buffer<double,1> ybuf(&y[0], N);
           
            // Compute inner product.
            q.submit([&] (auto &h) 
            {
                // accessors
                auto ax = xbuf.get_access<sycl::access::mode::read> (h);
                auto ay = ybuf.get_access<sycl::access::mode::read_write> (h);

                // loop         
                h.parallel_for(sycl::range<1>(N),  [=](sycl::id<1> i) {
                    ay[i] = alpha * ax[i] + ay[i];
                });
            }).wait();
            // <- buffers destroyed and memory synchronized!
        }
    };

}

#ifdef USE_SYCL
void run_test_sycl(int niter, int c)
{
    std::cout << "SYCL" << std::endl;
    SyclFixture f(niter, 2000*2000, c);
    std::cout << "Setup:" << std::endl;
    f.SetUp();
     std::cout << "Test:" << std::endl;
    PerformanceTestVector<mysycl::mvops>(f.test,f.niter, f.n);
    std::cout << "Teardown:" << std::endl;
    f.TearDown();
    std::cout << "done" << std::endl;
}
#endif