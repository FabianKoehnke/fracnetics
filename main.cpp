#include "include/Data.hpp"
#include "include/Population.hpp"
#include <chrono>

int main(){
    auto start = std::chrono::high_resolution_clock::now();
    Data main;
    /**
     * Initializing the population
     */
    Population population(123,1,5,5); 
    for(int i=0; i<population.individuals.size(); i++){
        auto net = population.individuals[i];
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Network: " << i << std::endl;
        std::cout << "---------------------------------" << std::endl;
        std::vector<Node> nodes = net.innerNodes;
        std::cout << "type: " << net.startNode.type << " id: " << net.startNode.id << std::endl;
        std::cout << "edges: " << net.startNode.edges[0] << std::endl;
        for(const auto& n : nodes){
            std::cout << "type: " << n.type << " id: " << n.id << std::endl;
            std::cout << "edges " << "(" << n.edges.size() << "): ";
            for(auto& ed : n.edges){
                std::cout << ed << " ";
            }
            std::cout << std::endl;

            std::cout << "boundaries" << "(" << n.boundaries.size() << "): ";
            for(auto& b: n.boundaries){
                std::cout << b << " ";
            }
            std::cout << std::endl;
 
        }
    }
 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "done in:" << duration.count() << "sek. \n"; 
    return 0;
}
