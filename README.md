# Benchmark suite for BLAS routines

[![Test Coverage](https://api.codeclimate.com/v1/badges/4574f8cee11c1e3a82aa/test_coverage)](https://codeclimate.com/github/anaegel/benchmark-matvec/test_coverage)

(c) Arne Naegel, Goethe University Frankfurt 

Aufruf:

```
./test-XYZ <nVectors> <someDouble>
```

 
a) Experimentieren Sie mit verschiedenen Optimierungen. Als Guideline:
- O0: Keine Optimierungen
- O1: Kleiner Code
- O2: Schnellster Code
- O3: Maximale Optimierung (ggf. Code-Umstellungen)

[Erläuterungen für gcc](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) 

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
