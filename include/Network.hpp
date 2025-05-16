#ifndef NETWORK_HPP
#define NETWORK_HPP
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include "Node.hpp"
#include <iostream>
/**
 * @class Network
 *
 * @brief This class defines the GNP graph.
 *
 * @param seed (const unsigned int): sets the seed
 * @param jn (unsigned int): number of initial judgment nodes
 * @param jnf (unsigned int): number of judgment node functions 
 * @param pn (unsigned int): number of initial processing nodes
 * @param pnf (unsigned int): number of processing node funcions
 *
 */

class Network {
    public:
        const unsigned int seed;
        unsigned int nn;
        unsigned int jn;
        unsigned int jnf;
        unsigned int pn;
        unsigned int pnf;
        std::vector<Node> innerNodes;
        Node startNode;
        float fitness = 0;

        Network(
                unsigned int _seed,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf
                ):
            seed(_seed),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            startNode(seed,0,_jn+_pn,"S",-1) // init start node
    {
        nn = pn+jn;
        for(int i=0; i<jn; i++){ // init judgment nodes 
            innerNodes.push_back(Node(seed+i,i,nn,"J",i%jnf));
        }
        for(int i=jn; i<jn+pn; i++){ // init procesing nodes 
            innerNodes.push_back(Node(seed+i,i,nn,"P",i%pnf));
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
                int dMax,
                int penalty
                ){
            int currentNodeID = startNode.edges[0];
            int dec;
            float correct = 0;
            for(int i=0; i<dt.size(); i++){
                int  dSum = 0; //to prevent dead-looks 
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
                }
                if (dSum >= dMax){
                    break;
                    correct /= penalty;
                }
            }
            fitness = correct / dt.size();
        }

};

#endif
