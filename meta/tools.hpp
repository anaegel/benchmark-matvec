#pragma once

#include <stdlib.h>

template <typename TVector>
void SetRandom(size_t n, int c, TVector &x)
{
	 for (size_t i=0; i<n; ++i)
	 { x[i] = 1.0*i*c; }
}

template <typename TVector>
void SetValue(size_t n, double c, TVector &x)
{
	 for (size_t i=0; i<n; ++i)
	 { x[i] = c; }
}



const int myAllocSize=64;

struct StdArrayAllocator
{
	typedef double* TVector;

	static void allocate_vector (size_t n, TVector &p)
	{

        #ifdef _WIN32
            p = (double*) _aligned_malloc(n*sizeof(double), myAllocSize);
        #else
           p = (double*) aligned_alloc(myAllocSize, n*sizeof(double));
           // p = new double[n];
        #endif
        //p = (double*) _mm_alloc( n * sizeof(double), myAllocSize);
    }


	static void  deallocate_vector(TVector &p)
	{
         #ifdef _WIN32
             _aligned_free(p);
        #else
            free(p);
            // delete p;
        #endif

        //_mm_free(p);
    }
};
