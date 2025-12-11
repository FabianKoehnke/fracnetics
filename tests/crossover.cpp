#include "../include/Population.hpp"

int main(){

    Population population(
            123, // seed 
            100, // number of networks
            5, // number of judgment nodes (jn)
            5, // number of jn functions 
            5, // number of processing nodes (pn)
            5, // number of pn functions
            false
            ); 

    return 0;
}
