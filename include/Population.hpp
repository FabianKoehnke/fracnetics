#ifndef POPULATION_HPP
#define POPULATION_HPP
#include "Network.hpp"

/**
 * @class Population 
 * @brief Defines the whole population.
 *
 * @param seed (unsigned int): set seed 
 * @param ni (unsigned int): number of individuals
 * @param jn (unsigned int): number of judgment nodes 
 * @param jnf (unsigned int): number of judgment node functions 
 * @param pn (unsigned int): number of processing nodes   
 * @param pnf (unsigned int): number of processing node funcions
 *
 */
class Population {
    public:
        const unsigned int seed;
        const unsigned int ni;
        unsigned int jn;
        unsigned int jnf;
        unsigned int pn;
        unsigned int pnf;
        std::vector<Network> individuals;

        Population(
                const unsigned int _seed,
                const unsigned int _ni,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf
                ):
            seed(_seed),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(seed,jn,jnf,pn,pnf));
        }
    }

        void setAllNodeBoundaries(std::vector<float>& minF, std::vector<float>& maxF){
            for(auto& network : individuals){
               for(auto& node : network.innerNodes){
                   if(node.type == "J"){
                       node.setEdgesBoundaries(minF[node.f], maxF[node.f]);
                   }
               } 
            }
        }

        void callFitness(std::vector<std::vector<float>> dt, std::vector<int>& yIndices, std::vector<int>& XIndices){
            for (auto& network : individuals){
                network.fitAccuracy(dt, yIndices, XIndices);
            }
        }

};

#endif 
