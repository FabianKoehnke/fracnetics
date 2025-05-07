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
            type(_type)
                
            {   
                if (type == "S") { // node is a start node 
                    // TODO: init edges
                    std::cout << "Start Node\n"<< std::endl;
                } else if (type == "P") { // node is a judgment node
                    std::cout << "Processing Node" << std::endl;
                } else if (type == "J") { // node is a processing node
                    std::cout << "Judgment Node" << std::endl;
                }
                edges = setEdges();

            }

        std::vector<int> setEdges(){
            std::mt19937 generator(seed); 
            std::uniform_int_distribution<int> distribution(0, this->nn-1);
            int randomInt = distribution(generator);

            std::cout << "random number:" << randomInt << std::endl;

            std::vector<int> edges(this->nn);
            for(int i=0; i<this->nn; i++){
                edges[i]=i;
            }
            std::shuffle(edges.begin(), edges.end(), generator);
            return std::vector<int>(edges.begin(), edges.begin()+randomInt);
        }
};
#endif
