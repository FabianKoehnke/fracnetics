#ifndef POPULATION_HPP
#define POPULATION_HPP
#include <vector>
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include <random>
#include <unordered_set>
#include <utility>
#include <cmath>
#include "Network.hpp"
#include "GymnasiumWrapper.hpp"
#include "PrintHelper.hpp"

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
        bool fractalJudgment;
        std::vector<Network> individuals;
        float bestFit;
        std::vector<int> indicesElite;
        float meanFitness = 0;
        float minFitness = std::numeric_limits<float>::max();

        Population(
                int seed,
                const unsigned int _ni,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf,
                bool _fractalJudgment
                ):
            generator(std::make_shared<std::mt19937_64>(seed)),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(generator,jn,jnf,pn,pnf,fractalJudgment));
        }
    }

        /*
         * @fn setAllNodeBoundaries
         * @brief call setEdgesBoundaries for each node of each individual 
         * @param minF (std::vector<float>&): min values of all features 
         * @param maxF (std::vector<float>&): max values of all features 
         */
        void setAllNodeBoundaries(std::vector<float>& minF, std::vector<float>& maxF){
            for(auto& network : individuals){
               for(auto& node : network.innerNodes){
                   if(node.type == "J"){
                       if(fractalJudgment == true){
                           node.productionRuleParameter = randomParameterCuts(node.k_d.first-1, generator);
                           std::vector<float> fractals = fractalLengths(node.k_d.second, sortAndDistance(node.productionRuleParameter));
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f], fractals);
                       }else {
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f]);
                       }
                   }
               } 
            }
        }

        /* @fn callFitness
         * @brief call the fitness for each individual
         * @note stores the bestFit of the population in member bestFit
         *
         * @param X (std::vector<std::vector<int>>&) : X of data table (features) 
         * @param y (std::vector<int>&) : y of data table (target values) 
         * @param dMax (int) : maximal judgments (delay) until next decision 
         * @param penalty (int) : devisor on fitness after exceeding maximal judgments
         * @param type (std::string) : name of the fitness function:
         *  - accuracy
         *  - cartpole
         * @param maxConsecutiveP (int) : maximal number of n consecutive processing nodes
         */
        void callFitness(
                std::vector<std::vector<double>>& X,
                std::vector<double>& y,
                int dMax,
                int penalty,
                std::string type,
                int maxConsecutiveP,
                GymEnvWrapper& env,
                int steps = 0
                ){
            bestFit = std::numeric_limits<float>::lowest();
            for (auto& network : individuals){
                if(type == "accuracy"){
                    network.fitAccuracy(X, y, dMax, penalty); 
                }else if (type == "cartpole") {
                    network.fitCartpole(dMax, penalty, 500, maxConsecutiveP);
                }else if (type == "gymnasium") {
                    network.fitGymnasium(
                            env,
                            dMax, 
                            penalty, 
                            steps, 
                            maxConsecutiveP);
                }

                if(network.fitness > bestFit){
                    bestFit = network.fitness;
                }
            }
        }

        /*
         * @fn tournamentSelection
         * @brief runs tournament selection and sets the new population
         * @param N (int): tournament size
         * @param E (int): size of elite
         */
        void tournamentSelection(int N, int E){
            std::vector<Network> selection;
            std::unordered_set<int> tournament;
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
            meanFitness = 0;
            minFitness = std::numeric_limits<float>::max();

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
                meanFitness += individuals[indexBestIndTournament].fitness;
                if (individuals[indexBestIndTournament].fitness < minFitness) {
                    minFitness = individuals[indexBestIndTournament].fitness;
                }
            }
            setElite(E, individuals, selection);
            individuals = std::move(selection);
            meanFitness /= individuals.size();
        }

        /*
         * @fn setElite
         * @brief stores the elite in given selection 
         * @param E (int): number of elite 
         * @param individuals (std::vector<Network>)
         * @param selection (std::vector<Network>&)
         */
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
                indicesElite.push_back(selection.size()); // because auf push_back of elite the index is the old size
                selection.push_back(individuals[eliteIndex]);
                individuals.erase(individuals.begin()+eliteIndex);
                counter += 1;
            }
        }

        /*
         * @fn callEdgeMutation
         * @brief call edgeMutation for each node in and each network (individual)
         * @param probInnerNodes (float): probability of changing inner nodes (jn and pn)
         * @param probStartNode (float): probability of changing the start node
         */
        void callEdgeMutation(float probInnerNodes, float probStartNode){
            for(int i=0; i<individuals.size(); i++){
                if(std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()){// preventing elite
                    for(auto& node : individuals[i].innerNodes){
                        node.edgeMutation(probInnerNodes, individuals[i].innerNodes.size());
                    }
                    individuals[i].startNode.edgeMutation(probStartNode, individuals[i].innerNodes.size());
                 }
             }
        }

        /*
         * struct for parameters used by edge mutations
         */
        struct additionalMutationParam{
            int networkSize = -1;
        };

         /*
         * @fn applyBoundaryMutation 
         * @brief apply the pased boundary mutation on each judgment node 
         * @param Func&& func (template function)
         */
        template <typename FuncMutation>
        void applyBoundaryMutation(FuncMutation&& func) {
            for (int i = 0; i < individuals.size(); ++i) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()) {
                    additionalMutationParam amp;
                    amp.networkSize = individuals[i].innerNodes.size();
                    for (auto& node : individuals[i].innerNodes) {
                        if (node.type == "J") {
                           func(node, amp); 
                        }
                    }
                }
            }
        }

        /*
         * @fn callBoundaryMutationUniform
         * @brief call method for boundaryMutationUniform
         * @param probability (const float): probability of mutate an edge
         */
        void callBoundaryMutationUniform(const float probability){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){ 
                node.boundaryMutationUniform(probability);
            });
        }
       
        /*
         * @fn callBoundaryMutationNormal 
         * @brief call method for boundaryMutationNormal
         * @param probability (const float): probability of mutate an edge
         * @param (const float): sigma of the distribution
         */
        void callBoundaryMutationNormal(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
                node.boundaryMutationNormal(probability, sigma);
            });
        }

        /*
         * @fn callBoundaryMutationEdgeSizeDependingSigma 
         * @brief call method for boundaryMutationNormal with sigmas depending on network size
         * @param probability (const float): probability of mutate an edge
         * @param (const float): sigma of the distribution
         */
        void callBoundaryMutationNetworkSizeDependingSigma(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam& amp){
                float sigmaNew = sigma * (1/log(amp.networkSize));
                node.boundaryMutationNormal(probability, sigmaNew);
            });
        }

        /*
         * @fn callBoundaryMutationEdgeSizeDependingSigma 
         * @brief call method for boundaryMutationNormal with sigmas depending on number of edges of judgment node 
         * @param probability (const float): probability of mutate an edge
         * @param (const float): sigma of the distribution
         */
        void callBoundaryMutationEdgeSizeDependingSigma(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
                float sigmaNew = sigma * (1/log(node.edges.size()));
                node.boundaryMutationNormal(probability, sigmaNew);
            });
        }
        
        /*
         * @fn callBoundaryMutationFractal
         * @brief call function for boundaryMutationFractal 
         * @param probability (float): probability of mutate an edge
         * @param minF (std::vector<float>&): min values of all features 
         * @param maxF (std::vector<float>&): max values of all features 
         */
        void callBoundaryMutationFractal(const float probability, std::vector<float> minF, std::vector<float> maxF){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
                node.boundaryMutationFractal(probability, minF, maxF);
            });
        }

        /*
         * @fn crossover
         * @brief exchange the nodes 
         * @note rules because of callAddDelNodes:
         *  - just change with node indices of {1,...,min(na,nb)}, where
         *    na and nb are the node number of individual a nd b 
         *  - change "dangling" edges (edges pointing to no node) randomly
         *  
         *  @param probability (float): probability of changing nodes
         */
        void crossover(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            std::vector<unsigned int> inds;
            for(int i=0; i<individuals.size(); i++){
                inds.push_back(i);
            }
            std::shuffle(inds.begin(), inds.end(), *generator);
            for(int i=0; i<inds.size()-1; i+=2){ // for each individual
                if(std::find(indicesElite.begin(), indicesElite.end(), inds[i]) != indicesElite.end() ||
                    std::find(indicesElite.begin(), indicesElite.end(), inds[i+1]) != indicesElite.end()
                    ){ // preventing crossover for elite
                    continue;
                }
                auto& parent1 = individuals[inds[i]];
                auto& parent2 = individuals[inds[i+1]];
                int maxNodeNumbers = std::min(parent1.innerNodes.size(), parent2.innerNodes.size());

                for(int k=0; k<maxNodeNumbers-1; k++){ // for each node
                    bool result = distributionBernoulli(*generator);
                    if(result){
                        std::swap(parent1.innerNodes[k], parent2.innerNodes[k]);
                        // just check for "false edges" if parent is the smaller one (expensive)
                        if(parent1.innerNodes.size() < parent2.innerNodes.size()){
                            parent1.changeFalseEdges();
                        } else if (parent2.innerNodes.size() < parent1.innerNodes.size()) {
                            parent2.changeFalseEdges(); 
                        }
                    }
 
                }

            }

        }

        /*
         * @fn callAddDelNodes 
         * @brief call addDelNodes for each individual 
         * @param minF (std::vector<float>&): min values of all features 
         * @param maxF (std::vector<float>&): max values of all features 
         */
        void callAddDelNodes(std::vector<float>& minF, std::vector<float>& maxF){
            for(auto& ind : individuals){
                ind.addDelNodes(minF, maxF);

            }
        }

};

#endif 
