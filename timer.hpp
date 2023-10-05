#pragma once


// Einige defines.
#define GIGA (1024*1024*1024)
#define MEGA (1024*1024)

#define TIMERSTART(start) {\
    auto start=std::chrono::steady_clock::now();

#define TIMERSTOP(start, flop, mem) \
    auto end = std::chrono::steady_clock::now();\
    std::chrono::duration<double> elapsed_seconds = end-start;\
    std::cout << "elapsed time: " << elapsed_seconds.count() <<  "s, ";\
    std::cout << "transfer: " << (mem/GIGA)/elapsed_seconds.count() <<  " GB/s, ";\
    std::cout << "computing: " << (flop/GIGA)/elapsed_seconds.count() <<  " GFLOP/s";}
