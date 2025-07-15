#ifndef NODE_HPP
#define NODE_HPP
#include <utility>
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include <vector>
#include <string>
#include <random>
#include "Fractal.hpp"
#include <iostream>

/**
 * @class Node 
 *
 * @brief This class defines the node of the GNP graph.
 *
 * @param generator (std::shared_ptr<std::mt19937>): passes the generator for random values
 * @param id (unsigned int): node id 
 * @param type (string): node type ("S"- Start Node, "P" - Processing Node or "J" Judgment Node)
 * @param f (unsigned int): node function to select feature ("J") or give output ("P")
 *
 */

class Node {
    private:
        std::shared_ptr<std::mt19937_64> generator; 
    public:
        unsigned int id;
        std::string type;
        unsigned int f;
        std::vector<int> edges;
        std::vector<double> boundaries;
        std::vector<float> productionRuleParameter;
        std::pair<int, int> k_d;
        
        Node(
            std::shared_ptr<std::mt19937_64> _generator,
            unsigned int _id, 
            std::string _type,
            unsigned int _f
            ):
            generator(_generator),
            id(_id),
            type(_type),
            f(_f)
                
            {}

        /**
         * @fn setEdges
         *
         * @brief set edges (member) of the node given the number of nodes of the network (nn).
         * @note The number of outgoing edges are (no self-loops allowed):
         *  - between [2,nn-1] for Judgment Nodes and 
         *  - 1 for Processing and Start Nodes
         * @param type (std::string):
         *      - "J" = Judgment Node 
         *      - "P" = Processing Node 
         *      - "S" = Start Node 
         * @param nn (int): node number to controle the number of outgoing edges
         * @param k (int): number of outgoing edges of a judgment node. 
         *  If k=0, a random number of outgoing edges is choosen (default).
         */
        void setEdges(std::string type, int nn, int k=0){

            if (type == "J") {
                for(int i=0; i<nn; i++){
                    if(i != this->id){//prevents self-loop
                        edges.push_back(i);    
                    }
                } 
                std::uniform_int_distribution<int> distribution(2, nn-1);
                int randomInt = distribution(*generator);// sets a random number of outgoing edges
                std::shuffle(edges.begin(), edges.end(), *generator);
                if(k == 0){
                    edges = std::vector<int>(edges.begin(), edges.begin()+randomInt);
                }else{
                    edges = std::vector<int>(edges.begin(), edges.begin()+k);
                }
            } else if(type == "S" || type == "P"){
                bool noSelfLoop = false;
                while(noSelfLoop == false){// prevents self-loop
                    std::uniform_int_distribution<int> distribution(0, nn-1);
                    int randomInt = distribution(*generator);// set a random successor
                    if(randomInt != this->id){
                        edges = std::vector<int>{randomInt};
                        noSelfLoop = true;
                        }
                    }
                } else {
                edges = std::vector<int>{};
            }
        }

        /**
         *
         * @fn judge 
         *
         * @brief judgement of a judgment node given edges, boundaries and feature value.
         * @note using binary search for finding intervall.
         * 
         * @param v (double): feature value 
         *
         * @return index of edge (<int>)
         *
         */
        int judge(double v){
            
            if(v <= boundaries[0]){
                return 0;
            } else if(v >= boundaries.back()){
                return edges.size()-1;
            } else {// do binary search
                int minIndex = 0;
                int maxIndex = edges.size()-1;
                while(minIndex <= maxIndex){
                    int midIndex = minIndex + (maxIndex - minIndex) / 2;
                    if(v >= boundaries[midIndex] && v < boundaries[midIndex+1]){
                       return midIndex;
                    } else if(v < boundaries[midIndex]){
                        maxIndex = midIndex-1;
                    } else{
                        minIndex = midIndex+1;
                    }
                }
            }// end binary search  
            return -1; 
        }

        /** 
         *
         * @fn setEdgesBoundaries
         *
         * @brief set the intervall boundaries for each outgoing edge of a node.
         *
         * @param minf (minFeatureValue)
         * @param maxf (maxFeatureValue)
         *
         */
        void setEdgesBoundaries(float minf, float maxf, std::vector<float> lengths = {}){ 
           float sum = minf;
           float span;
           for(int i = 0; i<edges.size()+1; i++){
               if(lengths.size()==0){
                   span = (maxf - minf) / edges.size();
               }else {
                   span = (maxf - minf) * lengths[i];
               } 
               boundaries.push_back(sum);
               sum += span;
           }
        }

        /*
         * @fn edgeMutation
         * @brief change the edges of a network (individual)
         * @param propability (float): propability of changing an edge 
         * @param nn (int): number of node of the network 
         */
        void edgeMutation(float propability, int nn){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(auto& edge : edges){
                bool result = distributionBernoulli(*generator);
                if(result){ 
                    changeEdge(nn, edge);
                }
            }
        }

        /*
         * @fn changeEdge
         * @brief change the successor of an edge 
         * @param nn (int): number of node of the network 
         * @param edge (int&): successor of edge of
         */
        void changeEdge(int nn, int& edge){
            std::uniform_int_distribution<int> distributionUniform(0, nn-1);
            bool noSelf = false;
            while(noSelf == false){ // prevent self-loop and same edge
                int randomInt = distributionUniform(*generator);// sets a random number of outgoing edges
                if(randomInt != this->id && randomInt != edge){
                    edge = randomInt;
                    noSelf = true;
                }
            }
        }

         /*
         * @fn boundaryMutationUniform 
         * @brief mutate the boundaries by shifting them between intervals by a random drawn number from the uniform distribution.
         * @param propability (float)
         */
        void boundaryMutationUniform(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i=1; i<boundaries.size()-1; i++){ // only shift the inner boundaries
                bool result = distributionBernoulli(*generator);
                if(result){
                    std::uniform_real_distribution<float> distributionUniform(boundaries[i-1], boundaries[i+1]);
                    boundaries[i] = distributionUniform(*generator);

                }
            }
        }

        /*
         * @fn boundaryMutationFractal
         * @brief shifting productionRuleParameter randomly (uniform), recalculats fractalLengths and boundaries
         * @param propability (float)
         * @param minf (std::vector<float>): min values of features 
         * @param maxf (std::vector<float>): max values of features
         */
        void boundaryMutationFractal(float propability, std::vector<float>& minf, std::vector<float>& maxf){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i=1; i<productionRuleParameter.size()-1; i++){ // only shift the inner parameter: [0,shift,...,1]
                bool result = distributionBernoulli(*generator);
                if(result){
                    std::uniform_real_distribution<float> distributionUniform(productionRuleParameter[i-1], productionRuleParameter[i+1]);
                    productionRuleParameter[i] = distributionUniform(*generator); 
                    boundaries.clear();
                    std::vector<float> fractals = fractalLengths(k_d.second, sortAndDistance(productionRuleParameter));
                    setEdgesBoundaries(minf[f], maxf[f], fractals);
                }
                }
            }

        /*
         * @fn boundaryMutationNormal
         * @brief mutate the boundaries by shifting them between intervals by a random drawn number from the normal distribution.
         * @param propability
         * @param sigma
         */
        void boundaryMutationNormal(float propability, float sigma){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i = 1; i<boundaries.size()-1; i++){ // only shift the inner boundaries
                bool result = distributionBernoulli(*generator);
                if(result){
                    float mu = boundaries[i];
                    std::normal_distribution<float> distributionNormal(mu,sigma);
                    float newBoundary = distributionNormal(*generator);
                    if(newBoundary > boundaries[i-1] && newBoundary < boundaries[i+1]){ // preventing overlapping boundaries
                        boundaries[i] = newBoundary; 
                    }
                }
            }
        }

};
#endif
