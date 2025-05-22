#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <random>
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include "Node.hpp"
#include <iostream>
/**
 * @class Network
 *
 * @brief This class defines the GNP graph.
 *
 * @param generator (std::shared_ptr<std::mt19937>): passes the generator for random values
 * @param jn (unsigned int): number of initial judgment nodes
 * @param jnf (unsigned int): number of judgment node functions 
 * @param pn (unsigned int): number of initial processing nodes
 * @param pnf (unsigned int): number of processing node funcions
 *
 */

class Network {
    private:
        std::shared_ptr<std::mt19937_64> generator;
    public:
        unsigned int jn;
        unsigned int jnf;
        unsigned int pn;
        unsigned int pnf;
        std::vector<Node> innerNodes;
        Node startNode;
        float fitness = std::numeric_limits<float>::lowest();

        Network(
                std::shared_ptr<std::mt19937_64> _generator,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf
                ):
            generator(_generator),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            startNode(generator,0,"S",-1) // init start node
            
    {
        startNode.setEdges("S", jn+pn);
        std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
        for(int i=0; i<jn; i++){ // init judgment nodes 
            int randomInt = distributionJNF(*generator);
            innerNodes.push_back(Node(
                        generator, 
                        i, // node id 
                        "J", // node type 
                        randomInt // node function
                        ));
            innerNodes.back().setEdges("J", pn+jn);
        }
        std::uniform_int_distribution<int> distributionPNF(0, pnf-1);
        for(int i=jn; i<jn+pn; i++){ // init procesing nodes 
            int randomInt = distributionPNF(*generator);
            innerNodes.push_back(Node(
                        generator, 
                        i, // node id 
                        "P", // node type 
                        randomInt // node function
                        ));
            innerNodes.back().setEdges("P", jn+pn);
        }
        
    }

        /**
         * @fn fitAccuracy
         * @brief executes transition path and calculates the accuracy.
         * @param dt (std::vector<std::vector<float>>& dt) : data table
         * @param yIndices (std::vector<int>&) : indices to select y values 
         * @param XIndices (std::vector<int>&) : indices to select X valaues (features)
         * @param dMax (int) : maximal judgments until next decision
         * @param penalty (int) : devisor on fitness after exceeding maximal judgments
         */
        void fitAccuracy(
                std::vector<std::vector<float>>& dt, 
                std::vector<int>& yIndices, 
                std::vector<int>& XIndices,
                int& dMax,
                int& penalty
                ){
            int currentNodeID = startNode.edges[0];
            int dec;
            float correct = 0;
            for(int i=0; i<dt.size(); i++){
                int  dSum = 0; // to prevent dead-looks 
                if (innerNodes[currentNodeID].type == "P"){
                    dec = innerNodes[currentNodeID].f;
                    if(dec == dt[i][yIndices[0]]){
                        correct += 1;
                    }
                    currentNodeID = innerNodes[currentNodeID].edges[0];
                } else if (innerNodes[currentNodeID].type == "J"){
                    while(innerNodes[currentNodeID].type == "J"){
                        float v = dt[i][XIndices[innerNodes[currentNodeID].f]];
                        currentNodeID = innerNodes[currentNodeID].edges[innerNodes[currentNodeID].judge(v)];
                        dSum += 1;
                        if (dSum >= dMax){
                            break;
                        }
                    }
                
                    dec = innerNodes[currentNodeID].f;
                    if(dec == dt[i][yIndices[0]]){
                        correct += 1;
                    }
                }
                if (dSum >= dMax){
                    break;
                    correct /= penalty;
                }
            }
            fitness = correct / dt.size();
        }

        void changeFalseEdges(){
            for(auto& node : innerNodes){
                for(auto& edge : node.edges){
                    if(edge > node.nn-1){ // edge as no successor node -> set new one
                        bool noSelfLoop = false;
                        while(noSelfLoop == false){ // prevents self-loop
                            std::uniform_int_distribution<int> distribution(0, node.nn-1);
                            int randomInt = distribution(*generator);
                            if(randomInt != node.id){
                                edge = randomInt;
                                noSelfLoop = true;
                            }
                        }

                    }
                }
            }
        }

};

#endif
