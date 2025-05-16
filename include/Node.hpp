#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <random>

/**
 * @class Node 
 *
 * @brief This class defines the node of the GNP graph.
 *
 * @param seed (const unsigned int): set random values (edge initialization)
 * @param id (unsigned int): node id 
 * @param nn (unsigned int): number of nodes of the network (edge initialization)
 * @param type (string): node type ("S"- Start Node, "P" - Processing Node or "J" Judgment Node)
 * @param f (unsigned int): node function to select feature ("J") or give output ("P")
 *
 */

class Node {
    public:
        const unsigned int seed;
        unsigned int id;
        unsigned int nn;
        std::string type;
        unsigned int f;
        std::vector<int> edges;
        std::vector<double> boundaries;
        
        Node(
            const unsigned int _seed, 
            unsigned int _id, 
            unsigned int _nn, 
            std::string _type,
            unsigned int _f
            ):
            seed(_seed),
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
            std::mt19937 generator(seed); 

            if (type == "J") {
                for(int i=0; i<this->nn; i++){
                    if(i != this->id){//prevents self-loop
                        edges.push_back(i);    
                    }
                } 
                std::uniform_int_distribution<int> distribution(2, this->nn-1);
                int randomInt = distribution(generator);// sets a random number of outgoing edges
                std::shuffle(edges.begin(), edges.end(), generator);
                edges = std::vector<int>(edges.begin(), edges.begin()+randomInt);
            } else if(type == "S" || type == "P"){
                bool noSelfLoop = false;
                while(noSelfLoop == false){// prevents self-loop
                    std::uniform_int_distribution<int> distribution(0, this->nn-1);
                    int randomInt = distribution(generator);// set a random successor
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

};
#endif
