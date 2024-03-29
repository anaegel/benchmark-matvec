
// Include SYCL.

#ifdef USE_SYCL

#include <CL/sycl.hpp>
using namespace cl;

// Fixtures.
#include "../meta/tools.hpp"
#include "../meta/fixtures.hpp"
#include "../meta/tests.hpp"




// StdArrayAllocator is aligned.
struct SyclFixture : public Fixture<StdArrayAllocator>
{
    typedef Fixture<StdArrayAllocator> fixture_type;

    SyclFixture(int niter, size_t n, int c)
    : fixture_type(niter, n, c) {}

    void SetUp()
    { fixture_type::SetUp(); }

    void TearDown()
    { fixture_type::TearDown(); }

};



// SYCL interface.
namespace mysycl {

    struct mvops {
    	static const int one = 1;
        template <class TVector>
        static double dot(const int N, const TVector &x, const TVector &y)
        {
        	sycl::queue q;
            // std::cout << "dot:" << q.get_device().get_info<sycl::info::device::name>() << std::endl;

           	// Buffers re-use existing memory.
            sycl::buffer<double,1> xbuf(&x[0], N);
            sycl::buffer<double,1> ybuf(&y[0], N);

            // Initialize zero.
            double dotResult = 0.0;
            sycl::buffer<double,1> dbuf(&dotResult, 1);

            // Compute inner product.
            q.submit([&] (auto &h) {
                auto ax = xbuf.get_access<sycl::access::mode::read> (h);
                auto ay = ybuf.get_access<sycl::access::mode::read> (h);

                auto asum = dbuf.get_access<sycl::access::mode::read_write> (h);
                auto red = sycl::reduction(asum, sycl::plus<double>());

                h.parallel_for(sycl::range<1>(N), red,
                	[=](sycl::id<1> i, auto &sum) { sum += ax[i] * ay[i]; }
                );
            }).wait();

           return dotResult;
        }

        template <class TVector>
        static double norm2(const int N, const TVector &x)
        {
        	sycl::queue q;
        	sycl::buffer<double,1> xbuf(&x[0], N);

        	// Initialize zero.
        	double dotResult = 0.0;
        	sycl::buffer<double,1> dbuf(&dotResult, 1);

        	// Compute norm.
        	q.submit([&] (auto &h) {
        		auto ax = xbuf.get_access<sycl::access::mode::read> (h);
        		auto asum = dbuf.get_access<sycl::access::mode::read_write> (h);
        		auto red = sycl::reduction(asum, sycl::plus<double>());

        		h.parallel_for(sycl::range<1>(N), red,
        				[=](sycl::id<1> i, auto &sum) { sum += ax[i] * ax[i]; }
        		);
        	}).wait();

        	return dotResult;
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


#endif

void run_test_sycl(int niter, int c)
{
#ifdef USE_SYCL
    SyclFixture f(niter, 2000*2000, c);
    f.SetUp();

    //std::cout << "Test:" << std::endl;
    UnitTest_BLAS_Level1<mysycl::mvops>(f.test, f.niter, f.n);
    PerfTest_BLAS_Level1<mysycl::mvops>(f.test,f.niter, f.n);

    //std::cout << "Teardown:" << std::endl;
    f.TearDown();
    //std::cout << "done" << std::endl;
#endif
}
