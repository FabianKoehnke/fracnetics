#ifndef POPULATION_HPP
#define POPULATION_HPP

#include "Network.hpp"

/**
 * @class Population 
 * @brief Defines the whole population.
 *
 * @param ni (unsigned int): number of individuals
 *
 */
class Population {
    public:
        unsigned int seed;
        unsigned int ni;
        unsigned int jn;
        unsigned int pn;
        std::vector<Network> individuals;

        Population(
                unsigned int _seed,
                unsigned int _ni,
                unsigned int _jn,
                unsigned int _pn
                ):
            seed(_seed),
            ni(_ni),
            jn(_jn),
            pn(_pn)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(seed,jn,pn));
        }
    }
};

#endif 
