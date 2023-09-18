#pragma once

struct EigenVectorAllocator
{
	typedef Eigen::VectorXd TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.resize(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};


// #ifdef USE_EIGEN
namespace myeigen {
struct mvops {
    template <class TVector>
    static double dot(const int N, const TVector &x, const TVector &y)
    { return x.dot(y); }

    template <class TVector>
    static double norm2(const int N, const TVector &x)
    { return x.dot(x); }

    template <class TVector>
    static void axpy(const int N, double alpha, const TVector &x, TVector &y)
    { y += alpha*x; }
};
}
