#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "Node.hpp"

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
            startNode(seed,0,_jn+_pn,"S",-1)
    {
        nn = pn+jn;
        for(int i=0; i<jn; i++){
            innerNodes.push_back(Node(seed+i,i+1,nn,"J",i%jnf));
        }
        for(int i=jn; i<jn+pn; i++){
            innerNodes.push_back(Node(seed+i,i+1,nn,"P",i%pnf));
        }
        
    }

};

#endif
