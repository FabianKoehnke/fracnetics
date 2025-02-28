#include "include/Data.hpp"
#include <chrono>

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    Data main;
    std::vector<std::vector<float > > data;
    data = main.readCSV("data/BTCUSD-1m-2025-01.csv");
    std::cout << "done reading csv file \n";

    double sum;
    for (int k = 0; k < 10000; ++k){

        for (int i = 0; i < data.size(); ++i){
        sum += data[i][0];
        //std::cout << "k = " << k;
    }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "sum = " << sum;
    std::cout << "done in:" << duration.count() << "sek. \n"; 
    return 0;
}