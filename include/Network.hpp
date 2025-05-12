#ifndef NETWORK_HPP
#define NETWORK_HPP

#include<string>
#include "Node.hpp"

/**
 * @class Network
 *
 * @brief This class defines the GNP graph.
 *
 */

class Network {
    public:
        unsigned int seed;
        unsigned int nn;
        unsigned int jn;
        unsigned int pn;
        std::vector<Node> innerNodes;
        Node startNode;

        Network(
                unsigned int _seed,
                unsigned int _jn,
                unsigned int _pn
                ):
            seed(_seed),
            jn(_jn),
            pn(_pn),
            startNode(seed,0,_jn+_pn,"S")
    {
        nn = pn+jn;
        for(int i=1; i<jn+1; i++){
            innerNodes.push_back(Node(seed+i,i,nn,"J"));
        }
        for(int i=1+jn; i<jn+1+pn; i++){
            innerNodes.push_back(Node(seed+i,i,nn,"P"));
        }
        
    }

};

#endif
