# Benchmark suite for BLAS routines

[![Test Coverage](https://api.codeclimate.com/v1/badges/4574f8cee11c1e3a82aa/test_coverage)](https://codeclimate.com/github/anaegel/benchmark-matvec/test_coverage)

(c) Arne Naegel, Goethe University Frankfurt 

Aufruf:

```
./test-XYZ <nVectors> <someDouble>
```

## Compiler settings 

### Optimization level 
Experimentieren Sie mit verschiedenen Optimierungen. Als Guideline:
- O0: Keine Optimierungen
- O1: Kleiner Code
- O2: Schnellster Code
- O3: Maximale Optimierung (ggf. Code-Umstellungen)

[Erläuterungen für gcc](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) 

### Machine specific binaries
Weitere nützliche Flags (Compiler-spezifisch!): 
- Intel: -xHost
- clang/gcc: -march=native



## Support for OpenMP

Most compilers support multithreading using OpenMP. To include this in the tests, use 

```
OMP_NUM_THREADS=1 ./test-XYZ 3 0.1
```

OpenMP support is activated by 
```
#define USE_OPENMP
```
and should be detected in the build process automatically.

##  Testing BLAS libraries
BLAS libraries are activated by 
```
#define USE_BLAS
#define USE_CBLAS
```
This is included in the cmake build process. 

##  Testing MKL
The MKL-provided linear algebra is activated by 
```
#define USE_MKL_BLAS
```
This is included in the cmake build process. However, no additional BLAS should be tested.

##  Testing Eigen3
The linear algebra of Eigen3 is activated by
```
#define USE_EIGEN3
```
If Eigen3 is installed, this should be detected automatically by the cmake build process.

## New CMake feature flags
The project exposes optional CMake switches to enable/disable specific backends and integrations.
All flags are OFF by default. Use them when configuring CMake, for example:

```
cmake -S . -B build -DENABLE_KOKKOS=ON -DENABLE_EIGEN3=ON
cmake -S . -B build -DENABLE_OPENCL=ON
```

Supported flags:
- `ENABLE_KOKKOS` - enable Kokkos integration (default OFF)
- `ENABLE_EIGEN3` - enable Eigen3 (default OFF)
- `ENABLE_UG4` - enable UG4 integration (default OFF)
- `ENABLE_INTEL_SYCL` - enable Intel SYCL support (default OFF)
- `ENABLE_ADAPTIVECPP` - enable AdaptiveCPP SYCL support (default OFF)
- `ENABLE_RAJA` - enable RAJA (default OFF)
- `ENABLE_OPENCL` - enable OpenCL (default OFF)
- `ENABLE_APPLE_METAL` - enable Apple Metal (default OFF)

These flags only control whether CMake attempts to find and wire the corresponding packages; the build still requires the relevant libraries/toolchains to be available on the system.

##  Testing UG4
The module allows to test the linear algebra provided by UG4, if 
```
#define USE_UG4
```
is activated. This is done if, if UG4 is found in the path provided by the environment variable *UG4_ROOT*. 


If UG4 has not been installed:
* Download UG4 to a separate directory.
* Set 'UG4_ROOT' environment variable.

In any case, you should enable the experimental feature of the 'feature-openmp' branch:
```
cd $UG4_ROOT
git checkout feature-openmp
```
  
##  Implementation status

| Setup         | dot | norm2 | axpy | matmul | matmul_tranpose | Requirements   |
|---------------|-----|-------|------|--------|-----------------|----------------|
| Plain         | x   | x     | x    |        |                 |                |
| SIMD          | x   | x     | x    |        |                 | OpenMP compiler|
| SIMD + OpenMP | x   | x     | x    |        |                 | OpenMP compiler|
| USE_CBLAS     | x   | x     | x    |        |                 |                |
| USE_MKL       | x   | x     | x    |        |                 |                |
| USE_UG4       | x   | x     | x    | x      | x               | UG4            |
| USE_EIGEN3    | x   | x     | x    | x      | x               | Eigen3         |
| USE_SYCL      | exp | exp   | exp  |        |                 |                |
