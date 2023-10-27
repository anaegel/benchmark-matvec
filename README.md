# Entwurf einer Testsuite 


[![Test Coverage](https://api.codeclimate.com/v1/badges/4574f8cee11c1e3a82aa/test_coverage)](https://codeclimate.com/github/anaegel/benchmark-matvec/test_coverage)

Modellierung und Simulation 2
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


b) Via OpenMP unterstützen die Compiler bereits Multithreading. Experimentieren Sie mit```

```
OMP_NUM_THREADS=1 ./test-XYZ
```

##  Implementation status

| Setup         | dot | norm2 | axpy | matmul | matmul_tranpose |
|---------------|-----|-------|------|--------|-----------------|
| Plain         | x   | x     | x    |        |                 |
| SIMD          | x   | x     | x    |        |                 |
| SIMD + OpenMP | x   | x     | x    |        |                 |
| USE_CBLAS     | x   | x     | x    |        |                 |
| USE_MKL       | x   | x     | x    |        |                 |
| USE_UG4       | x   | x     | x    | x      | x               |
| USE_EIGEN3    | x   | x     | x    | x      | x               |
