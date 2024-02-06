#pragma once


#include <math.h>
#include <iostream>



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



//! This fixture provides a 2D finite difference discretization.
template <typename TAllocator, typename TVector = typename TAllocator::TVector, typename TMatrix= typename TAllocator::TMatrix>
struct MatrixVectorFixture {

	MatrixVectorFixture(size_t ncells)
	: ncells(ncells), n(ncells*ncells) {}

    void SetUp()
    {
    	TAllocator::allocate_vector(n, x);
    	TAllocator::allocate_vector(n, b);

    	A = TAllocator::create_matrix(ncells);
    }


    void TearDown()
    {
    	TAllocator::deallocate_vector(x);
    	TAllocator::deallocate_vector(b);

    	delete A;

    }

    ~MatrixVectorFixture() {}

    const size_t ncells;			// Size of test vector
    const size_t n;					// Size of test vector

    TVector x;
    TVector b;
	TMatrix *A;

};
