#ifndef PRINTHELPER_HPP
#define PRINTHELPER_HPP

#include <sys/resource.h>
#include <iostream>

inline void printMemoryUsage(){
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory used: " << usage.ru_maxrss << " KB" << std::endl;
}

inline void printLine(){
    std::cout << "---------------------------------" << std::endl;
}
#endif
