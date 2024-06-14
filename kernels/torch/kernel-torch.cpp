// Pytorch
#include <torch/torch.h>


void SetRandom(size_t n, int c, torch::Tensor &x)
{ x.random_(); }

void SetValue(size_t n, double c, torch::Tensor &x)
{ x.fill_(c); }

// Fixtures (meta-programming)
#include "../../meta/fixtures.hpp"
#include "../../meta/tests.hpp"

auto options = torch::TensorOptions()
     .dtype(torch::kFloat32).device(torch::kMPS)  // Apple Silicon.
    //.dtype(torch::kFloat64)
    .layout(torch::kStrided)
    .requires_grad(false);


void PrintTorchTensorInfo(torch::Tensor &x)
{
    std::cout<< "dtype " << x.dtype() << ", " <<
                "layout " << x.layout() << ", " <<
                "device " << x.device() << ", " <<
                "req_grad " << x.requires_grad() << std::endl;
}



//! Allocating vectors
struct TorchVectorAllocator
{
	typedef torch::Tensor TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ 
        v = torch::zeros({static_cast<long long>(n)}, options);
        PrintTorchTensorInfo(v);
    }

	static void deallocate_vector (TVector &v)
	{ 
        v = torch::zeros({static_cast<long long>(1)}, options); 
    }
};





namespace mytorch {
struct mvops {

    // Level 1
    template <class TVector>
    static double dot(const int N, const TVector &x, const TVector &y)
    {  
        auto dot =torch::inner(x,y);
        return dot.template item<double>();   
        // return torch::inner(x,y).cpu().template item<double>(); 
     }

    template <class TVector>
    static double norm2(const int N, const TVector &x)
    {   
        auto dot =torch::norm(x);
        return dot.template item<double>();  
        // return dot.cpu().template item<double>(); 
    }

    template <class TVector>
    static void axpy(const int N, double alpha, const TVector &x, TVector &y)
    {   y=y.add(x, alpha);  }

    // Level 2
    template <class TVector, class TMatrix>
    static void matmul(const int N, TVector &y, const TMatrix &A, const TVector &x)
    {}

    template <class TVector, class TMatrix>
    static void matmul_transposed(const int N, TVector &y, const TMatrix &A, const TVector &x)
    {}

    template <class TVector, class TMatrix>
    static void matmul_add(const int N, TVector &y, const TMatrix &A, const TVector &x)
    {}
};
}


void run_test_torch(int niter, int c, size_t nentries)
{
    #ifdef APPLE
        std::cout << torch::mps::is_available();
        /* if (torch::mps::is_available()) {
          auto mps_device = torch::device(torch::DeviceType::kMPS);
        } */  
    #endif 

    torch::manual_seed(c);
   /* 
    torch::Tensor x = torch::zeros({static_cast<long long>(nentries)});
    PrintTorchTensorInfo(x);

    torch::Tensor y = torch::zeros({static_cast<long long>(nentries)}, options);
    PrintTorchTensorInfo(y);
    */
   
    Fixture<TorchVectorAllocator, TorchVectorAllocator::TVector> f(niter, NVECTOR, c);
	f.SetUp();    
    
    UnitTest_BLAS_Level1<mytorch::mvops>(f.test, f.niter, f.n);
    torch::mps::synchronize();
    PerfTest_BLAS_Level1<mytorch::mvops>(f.test, f.niter, f.n);

    f.TearDown();
    

}