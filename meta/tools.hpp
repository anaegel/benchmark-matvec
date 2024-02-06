#pragma once

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
