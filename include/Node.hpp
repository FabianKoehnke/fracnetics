#ifndef NODE_HPP
#define NODE_HPP
#define DEBUG_VAR(x) std::cout << #x << " = " << x << std::endl;

#include <vector>
#include <string>
#include <random>
#include <iostream>

/**
 * @class Node 
 *
 * @brief This class defines the node of the GNP graph.
 *
 * @param generator (std::shared_ptr<std::mt19937>): passes the generator for random values
 * @param id (unsigned int): node id 
 * @param nn (unsigned int): number of nodes of the network (edge initialization)
 * @param type (string): node type ("S"- Start Node, "P" - Processing Node or "J" Judgment Node)
 * @param f (unsigned int): node function to select feature ("J") or give output ("P")
 *
 */

class Node {
    private:
        std::shared_ptr<std::mt19937_64> generator; 
    public:
        unsigned int id;
        unsigned int nn;
        std::string type;
        unsigned int f;
        std::vector<int> edges;
        std::vector<double> boundaries;
        
        Node(
            std::shared_ptr<std::mt19937_64> _generator,
            unsigned int _id, 
            unsigned int _nn, 
            std::string _type,
            unsigned int _f
            ):
            generator(_generator),
            id(_id),
            nn(_nn),
            type(_type),
            f(_f)
                
            {   
                if (type == "S") { // node is a start node 
                    setEdges(type);
                } else if (type == "P") { // node is a judgment node
                    setEdges(type);
                } else if (type == "J") { // node is a processing node
                    setEdges(type);
                }
            }

        /**
         * @fn setEdges
         *
         * @brief set edges (member) of the node given the number of nodes of the network (nn).
         * @note The number of outgoing edges are:
         *  - between [1,nn-1] for Judgment Nodes and 
         *  - 1 for Processing and Start Nodes
         *
         */
        void setEdges(std::string type){

            if (type == "J") {
                for(int i=0; i<this->nn; i++){
                    if(i != this->id){//prevents self-loop
                        edges.push_back(i);    
                    }
                } 
                std::uniform_int_distribution<int> distribution(1, this->nn-1);
                int randomInt = distribution(*generator);// sets a random number of outgoing edges
                std::shuffle(edges.begin(), edges.end(), *generator);
                edges = std::vector<int>(edges.begin(), edges.begin()+randomInt);
            } else if(type == "S" || type == "P"){
                bool noSelfLoop = false;
                while(noSelfLoop == false){// prevents self-loop
                    std::uniform_int_distribution<int> distribution(0, this->nn-1);
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
        void setEdgesBoundaries(float minf, float maxf){
           double span = (maxf - minf) / edges.size();
           double sum = minf;
           for(int i = 0; i<edges.size()+1; i++){
               boundaries.push_back(sum);
               sum += span;
           }

        }

        void edgeMutation(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            std::uniform_int_distribution<int> distributionUniform(0, this->nn-1);
            for(auto& edge : edges){
                bool result = distributionBernoulli(*generator);
                
                if(result){
                    bool noSelf = false;
                    while(noSelf == false){ // prevent self-loop and same edge
                        int randomInt = distributionUniform(*generator);// sets a random number of outgoing edges
                        if(randomInt != this->id && randomInt != edge){
                            edge = randomInt;
                            noSelf = true;
                        }

                    }
                }
            }

        }

};
#endif
