#pragma once
#ifndef BENCHMARK_H_
#define BENCHMARK_H_
#endif

#include <Windows.h>
#include <iomanip>
//#include <time.h>

// CPU
#define prepare(x) QueryPerformanceFrequency(&(x))
#define timer(x) QueryPerformanceCounter(&(x))
#define total(op,ed) (((ed).QuadPart-(op).QuadPart)*1000000/freq.QuadPart)
#define output(method, t) std::cout<<std::left<<std::setw(5)<<method<<" time(us): "<<std::setw(8)<<t<<"     "<<std::endl
constexpr auto BASE = 1000000;
constexpr auto TOTALNUM = 100000;
constexpr auto COUNT = 100;




//// GPU
//clock_t clock_start, clock_end;
//double clock_diff_sec;