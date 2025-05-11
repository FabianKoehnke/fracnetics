#ifndef NODE_HPP
#define NODE_HPP

#include <vector>
#include <string>
#include <random>
#include <iostream>

/**
 * @class Node 
 *
 * @brief This class defines the node of the GNP graph.
 *
 * @param seed (const unsigned int): set random values (edge initialization)
 * @param id (unsigned int): node id 
 * @param nn (unsigned int): number of nodes of the network (edge initialization)
 * @param type (string): node type ("S"- Start Node, "P" - Processing Node or "J" Judgment Node)
 *
 */

class Node {
    public:
        const unsigned int seed;
        unsigned int id;
        unsigned int nn;
        std::string type;
        std::vector<int> edges;
        
        Node(
            const unsigned int _seed, 
            unsigned int _id, 
            unsigned int _nn, 
            std::string& _type
            ):
            seed(_seed),
            id(_id),
            nn(_nn),
            type(_type),
            edges(nn)
                
            {   
                if (type == "S") { // node is a start node 
                    // TODO: init edges
                    std::cout << "Start Node\n"<< std::endl;
                    setEdges(type);
                } else if (type == "P") { // node is a judgment node
                    std::cout << "Processing Node" << std::endl;
                    setEdges(type);
                } else if (type == "J") { // node is a processing node
                    std::cout << "Judgment Node" << std::endl;
                    setEdges(type);
                }

            }

        /**
         * @fn setEdges
         *
         * @brief set edges of the node given the number of nodes of the network (nn).
         * @note The number of outgoing edges are:
         *  - between [1,nn-1] for Judgment Nodes and 
         *  - 1 for Processing and Start Nodes
         *
         * @return edges (vector<int>)
         *
         */
        void setEdges(std::string type){
            std::mt19937 generator(seed); 
            std::uniform_int_distribution<int> distribution(0, this->nn-1);
            int randomInt = distribution(generator);

            if (type == "J") {
                for(int i=0; i<this->nn; i++){
                    edges[i]=i;
                }
                std::shuffle(edges.begin(), edges.end(), generator);
                edges = std::vector<int>(edges.begin(), edges.begin()+randomInt);
            } else if(type == "S" || type == "P"){
                edges = std::vector<int>{randomInt};
            } else {
                edges = std::vector<int>{};
            }
        }
};
#endif
