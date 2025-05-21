#ifndef POPULATION_HPP
#define POPULATION_HPP
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include <random>
#include <unordered_set>
#include <utility>
#include "Network.hpp"

/**
 * @class Population 
 * @brief Defines the whole population.
 *
 * @param generator (std::shared_ptr<std::mt19937>): passes the generator for random values
 * @param ni (unsigned int): number of individuals
 * @param jn (unsigned int): number of judgment nodes 
 * @param jnf (unsigned int): number of judgment node functions 
 * @param pn (unsigned int): number of processing nodes   
 * @param pnf (unsigned int): number of processing node funcions
 *
 */
class Population {
    private:
        std::shared_ptr<std::mt19937_64> generator;
    public:
        const unsigned int ni;
        unsigned int jn;
        unsigned int jnf;
        unsigned int pn;
        unsigned int pnf;
        std::vector<Network> individuals;
        float bestFit;
        std::vector<int> indicesElite;

        Population(
                std::shared_ptr<std::mt19937_64> _generator,
                const unsigned int _ni,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf
                ):
            generator(_generator),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(generator,jn,jnf,pn,pnf));
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

        void callFitness(
                std::vector<std::vector<float>> dt,
                std::vector<int>& yIndices,
                std::vector<int>& XIndices,
                int& dMax,
                int& penalty
                ){
            bestFit = std::numeric_limits<float>::lowest();
            for (auto& network : individuals){
                network.fitAccuracy(dt, yIndices, XIndices, dMax, penalty);

                if(network.fitness > bestFit){
                    bestFit = network.fitness;
                }
            }
        }

        void tournamentSelection(int N, int E){
            std::vector<Network> selection;
            std::unordered_set<int> tournament;
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);

            for(int i=0; i<individuals.size()-E; i++){
                float bestFitTournament = std::numeric_limits<float>::lowest();
                int indexBestIndTournament = 0;
                tournament.clear();

                while(tournament.size()<N){ // set the tournament
                    int randomInt = distribution(*generator);
                    tournament.insert(randomInt);
                }
                for(int k : tournament){
                   if(individuals[k].fitness > bestFitTournament){
                       bestFitTournament = individuals[k].fitness;
                       indexBestIndTournament = k;
                   } 
                }
                selection.push_back(individuals[indexBestIndTournament]);
            }
            setElite(E, individuals, selection);
            individuals = std::move(selection);
        }

        void setElite(int E, std::vector<Network> individuals, std::vector<Network>& selection){
            unsigned int counter = 0;
            unsigned int eliteIndex = 0;
            indicesElite.clear();
            while(counter<E){
                float eliteFit = individuals[0].fitness;
                for(int i=1; i<individuals.size(); i++){
                    if(individuals[i].fitness > eliteFit){
                        eliteFit = individuals[i].fitness;
                        eliteIndex = i;
                    }
                }
                indicesElite.push_back(eliteIndex+counter);
                selection.push_back(individuals[eliteIndex]);
                individuals.erase(individuals.begin()+eliteIndex);
                counter += 1;
            }
        }

        void callEdgeMutation(float probInnerNodes, float probStartNode){
            for(int i=0; i<individuals.size(); i++){
                if(std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end()){// preventing elite
                    for(auto& node : individuals[i].innerNodes){
                        node.edgeMutation(probInnerNodes);
                    }
                    individuals[i].startNode.edgeMutation(probStartNode);
                 }
             }
        }

        void crossOver(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            std::vector<unsigned int> inds;
            for(int i=0; i<individuals.size(); i++){
                inds.push_back(i);
            }
            std::shuffle(inds.begin(), inds.end(), *generator);
            for(int i=0; i<inds.size()/2; i+=2){
                auto& parent1 = individuals[i];
                auto& parent2 = individuals[i+1];
                int maxNodeNumbers = std::min(parent1.nn, parent2.nn);
                DEBUG_VAR(maxNodeNumbers)

                for(int k=0; k<maxNodeNumbers-1; k++){
                    bool result = distributionBernoulli(*generator);
                    if(result){
                        DEBUG_VAR(parent1.innerNodes.size())
                        DEBUG_VAR(parent2.innerNodes.size())
                        std::swap(parent1.innerNodes[k], parent2.innerNodes[k]);
                        // just check "false edges" if parent is smaller one and unequal to other parent (expensive).
                        if(parent1.nn < maxNodeNumbers && parent1.nn != parent2.nn){
                            parent1.changeFalseEdges();
                        } else if (parent2.nn < maxNodeNumbers && parent2.nn != parent1.nn) {
                            parent2.changeFalseEdges(); 
                        }
                    }
 
                }

            }

        }

};

#endif 
