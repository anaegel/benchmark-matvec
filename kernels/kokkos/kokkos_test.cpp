#include <Kokkos_Core.hpp>
#include <iostream>
#include "kernel-kokkos-simd.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    const int N = 1000;
    using Alloc = mykokkos::simd::VectorAllocator;
    using simd_t = Alloc::simd_t;
    using TV = Alloc::TVector;

    TV v;
    Alloc::allocate_vector(N, v);

    auto ptr = reinterpret_cast<uintptr_t>(v.data());
    std::cout << "allocate_vector ptr=" << (void*)ptr << " mod64=" << (ptr % 64) << " packs=" << v.extent(0) << "\n";

    // initialize packs (set every lane to 1.0)
    const auto npacks = v.extent(0);
    Kokkos::parallel_for("init_packs", npacks, KOKKOS_LAMBDA(const int i){
      simd_t tmp(1.0);
      v(i) = tmp;
    });

    double sum = 0.0;
    Kokkos::parallel_reduce("sum_packs", npacks, KOKKOS_LAMBDA(const int i, double &lsum){
      simd_t tmp = v(i);
      lsum += mykokkos::simd::horizontal_add<simd_t, double>(tmp);
    }, sum);

    std::cout << "pack-sum=" << sum << " expected=" << (double)N << "\n";

    Alloc::deallocate_vector(v);
  }
  Kokkos::finalize();
  return 0;
}
