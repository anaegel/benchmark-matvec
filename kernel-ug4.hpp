#pragma once

// #include "common/log.h"
#include "lib_algebra/cpu_algebra_types.h"


struct UG4Allocator
{
	typedef typename ug::CPUAlgebra::vector_type TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.create(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};


template <unsigned int bs>
struct UG4AllocatorBlock
{
	typedef typename ug::CPUBlockAlgebra<bs>::vector_type TVector;

	static void allocate_vector(size_t n, TVector &v)
	{ v.create(n); }

	static void deallocate_vector (TVector &v)
	{ v.resize(0); }
};



namespace myug4 {
struct mvops {

	// using  operations_vec.h
    template <class TVector>
    static double dot(const int N, const TVector &x, const TVector &y)
    {
    	// return x.dotprod(y);
    	return ug::VecProd(x,y);
    }

    template <class TVector>
    static double norm2(const int N, const TVector &x)

    {
    	return ug::VecNormSquared(x); // ug::VecProd(x,x);
    }

    template <class TVector>
    static void axpy(const int N, double alpha, const TVector &x, TVector &y)
    {
    	ug::VecScaleAdd(y, alpha, x, 1.0, y);
    }
};
}



// Some dummies.
namespace ug {

	LogAssistant& LogAssistant::instance() {}
}


void  ug_assert_failed() {}
