#ifndef NETWORK_HPP
#define NETWORK_HPP
#include <random>
#include <unordered_set>
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
        std::unordered_set<int> usedNodes; // ids of nodes

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
            usedNodes.clear();
            int currentNodeID = startNode.edges[0];
            usedNodes.insert(currentNodeID);
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
                    usedNodes.insert(currentNodeID);
                } else if (innerNodes[currentNodeID].type == "J"){
                    while(innerNodes[currentNodeID].type == "J"){
                        float v = dt[i][XIndices[innerNodes[currentNodeID].f]];
                        currentNodeID = innerNodes[currentNodeID].edges[innerNodes[currentNodeID].judge(v)];
                        usedNodes.insert(currentNodeID);
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
                std::uniform_int_distribution<int> distribution(0, innerNodes.size()-1);
                for(auto& edge : node.edges){
                    if(edge > innerNodes.size()-1){ // edge has no successor node -> set new one
                        node.changeEdge(innerNodes.size(), edge);
                    }
                }
            }
        }

        /*
         * @fn addDelNodes
         * @brief add and delete nodes to the individual (network)
         * @param minf (std::vector<float>&): min values of all features 
         * @param maxf (std::vector<float>&): min values of all features 
         */
        void addDelNodes(std::vector<float>& minf, std::vector<float>& maxf){ 
            std::bernoulli_distribution distributionBernoulliAdd(0.5);
            float pnRatio = static_cast<float>(pn) / static_cast<float>(innerNodes.size());
            std::bernoulli_distribution distributionBernoulliProcessingNode(pnRatio);
            bool resultAdd = distributionBernoulliAdd(*generator);
            for(int n=0; n<innerNodes.size(); n++){
                if(resultAdd && usedNodes.size() >= innerNodes.size()){// adding node
                    bool resultProcessingNode = distributionBernoulliProcessingNode(*generator);
                    if(resultProcessingNode){ // add processing node
                        std::uniform_int_distribution<int> distributionPNF(0, pnf-1);
                            int randomInt = distributionPNF(*generator);
                            innerNodes.push_back(Node(
                                        generator, 
                                        innerNodes.size(), // node id 
                                        "P", // node type 
                                        randomInt // node function
                                        ));
                            innerNodes.back().setEdges("P", innerNodes.size());
                            pn += 1;
                    }else{ // add judgment node 
                        std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
                            int randomInt = distributionJNF(*generator);
                            innerNodes.push_back(Node(
                                        generator, 
                                        innerNodes.size(), // node id 
                                        "J", // node type 
                                        randomInt // node function
                                        ));
                            innerNodes.back().setEdges("J", innerNodes.size());
                            innerNodes.back().setEdgesBoundaries(minf[randomInt], maxf[randomInt]);
                            jn += 1;
                    }
                    break;
                }else if(!resultAdd && 
                        innerNodes.size()-usedNodes.size() > 1 && 
                        std::find(usedNodes.begin(), usedNodes.end(), innerNodes[n].id) == usedNodes.end())
                {// deleting nodes
                    for(int i=0; i<innerNodes.size(); i++){ // for each node
                        if(innerNodes[i].id > innerNodes[n].id){
                            innerNodes[i].id -= 1; // set back node numbers for nodes greater deleted id 
                        }
                        for(int k=0; k<innerNodes[i].edges.size(); k++){ // for each edge
                            if(innerNodes[i].edges[k] > innerNodes[n].id){
                                innerNodes[i].edges[k] -= 1; // change edges to reset node ids 
                            }else if(innerNodes[i].edges[k] == innerNodes[n].id){ // change edge pointing to deleted node
                                innerNodes[i].changeEdge(innerNodes.size()-1, innerNodes[i].edges[k]);
                            }
                        }
                    } 
                    startNode.changeEdge(innerNodes.size()-1, startNode.edges[0]);
                    innerNodes.erase(innerNodes.begin()+innerNodes[n].id);
                }
            }
        }

};

#endif
