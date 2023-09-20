** Entwurf einer Testsuite **


![CodeClimate]([https://api.codeclimate.com/v1/badges/4574f8cee11c1e3a82aa/maintainability])

Modellierung und Simulation 2 - Winter 2021/22
(c) G-CSC, Uni Frankfurt

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
