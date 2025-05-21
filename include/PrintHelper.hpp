#ifndef PRINTHELPER_HPP
#define PRINTHELPER_HPP

#include <sys/resource.h>
#include <iostream>
#include <vector>
#include <string>

inline void printMemoryUsage(){
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Memory used: " << usage.ru_maxrss << " KB" << std::endl;
}

inline void printLine(){
    std::cout << "---------------------------------" << std::endl;
}

template <typename T>
inline void printVec(const std::vector<T>& vec, std::string name){
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

#endif
