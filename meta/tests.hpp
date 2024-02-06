#pragma once



#include "tools.hpp"		// SetRandom.
#include "timer.hpp"   		// Timing operations.

#define NCELLS 2000
#define NVECTOR 4000000




////////////////////////////////////////////////
// BLAS - Level 1
////////////////////////////////////////////////

// Unit tests.
template <typename F, typename V>
void UnitTest_BLAS_Level1(V vec, size_t niter, size_t n, size_t block_size=1)
{
	{

		// for (size_t i=0; i<n; ++i) { vec[0][i] = 1.0; vec[1][i] = 0.0;}
        SetValue(n, 1.0, vec[0]);
        SetValue(n, 0.0, vec[1]);

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
