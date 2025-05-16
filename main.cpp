#include "include/Data.hpp"
#include "include/Population.hpp"
#include "include/PrintHelper.hpp"
#include <chrono>

int main(){
    /**
     * Rading Data
     */
    auto start = std::chrono::high_resolution_clock::now();
    Data data;
    std::cout << "reading data" << std::endl;
    printMemoryUsage();
    data.readCSV("data/IRIS.csv");
    printMemoryUsage();
    std::cout << "data rows: " << data.dt.size() << std::endl;
    std::cout << "data columns: " << data.dt[0].size() << std::endl;
    printLine();
    data.columnSelector(std::pair<int, int> (5,6), std::pair<int, int> (1,5)); // set the indices of y and X
    data.minMaxFeatures(); // calculate the min and max values of X (for node boundaries)
    /**
     * Initializing the population
     */
    Population population(
            223, // seed
            1, // number of networks
            1, // number of judgment nodes (jn)
            4, // number of jn functions 
            3, // number of processing nodes (pn)
            3 // number of pn functions
            ); 
    population.setAllNodeBoundaries(data.minX,data.maxX);
    for(int i=0; i<population.individuals.size(); i++){
        auto& net = population.individuals[i];
        printLine(); 
        std::cout << "Network: " << i << std::endl;
        printLine(); 
        std::cout << "type: " << net.startNode.type << " id: " << net.startNode.id << std::endl;
        std::cout << "edges: " << net.startNode.edges[0] << std::endl;
        for(const auto& n : net.innerNodes){
            std::cout << "type: " << n.type << " id: " << n.id << " ";
            std::cout << "edges " << "(" << n.edges.size() << "): ";
            for(auto& ed : n.edges){
                std::cout << ed << " ";
            }

            std::cout << "boundaries" << "(" << n.boundaries.size() << "): ";
            for(auto& b: n.boundaries){
                std::cout << b << " ";
            }
            std::cout << std::endl;
 
        }
    }
    printLine(); 
    population.callFitness(data.dt, data.yIndices, data.XIndices);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "done in:" << duration.count() << "sek. \n"; 
    return 0;
}
