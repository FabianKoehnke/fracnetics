#include "include/Data.hpp"
#include "include/Population.hpp"
#include "include/PrintHelper.hpp"
#include <chrono>
#include <random>

int main(){
    /**
     * Parameter
     */
    auto generator = std::make_shared<std::mt19937_64>(52);
    float probEdgeMutationStartNode = 0.02;
    float probEdgeMutationInnerNodes = 0.02;
    float probCrossOver = 0.01;
    int generations = 1000;
    int nIndividuals = 1001;
    int tournamentSize = 2;
    int nElite = 1;
    int jn = 1;
    int jnf = 4;
    int pn = 2;
    int pnf = 2;
    int dMax = 10;
    int penalty = 2;
    int maxConsecutiveP = 2;
    /**
     * Rading Data
     */
    auto start = std::chrono::high_resolution_clock::now();
    Data data;
    std::cout << "reading data" << std::endl;
    printMemoryUsage();
    data.readCSV("data/cartpole.csv");
    printMemoryUsage();
    std::cout << "data rows: " << data.dt.size() << std::endl;
    std::cout << "data columns: " << data.dt[0].size() << std::endl;
    printLine();
    data.columnSelector(std::pair<int, int> (0,0), std::pair<int, int> (0,4)); // set the indices of y and X
    data.minMaxFeatures(); // calculate the min and max values of X (for node boundaries)
    printVec(data.minX, "minX");
    printVec(data.maxX, "miny");
    /**
     * Initializing the population
     */
    Population population(
            generator, // generator
            nIndividuals, // number of networks
            jn, // number of judgment nodes (jn)
            jnf, // number of jn functions 
            pn, // number of processing nodes (pn)
            pnf // number of pn functions
            ); 
    population.setAllNodeBoundaries(data.minX,data.maxX);
    printLine(); 
    std::cout << "start EA" << std::endl;
    std::vector<float> bestFitnessPerGeneration;
    for(int g=0; g<generations; g++){
        population.callFitness(data.dt, data.yIndices, data.XIndices, dMax, penalty, "cartpole", maxConsecutiveP);
        population.tournamentSelection(tournamentSize,nElite);
        population.callEdgeMutation(probEdgeMutationInnerNodes, probEdgeMutationStartNode);
        population.crossover(probCrossOver);
        population.callAddDelNodes(data.minX, data.maxX);
        std::cout << "Geneation: " << g << " BestFit: " << population.individuals[population.indicesElite[0]].fitness << std::endl;
        //std::cout << "Best Fit: " << population.individuals[population.indicesElite[0]].fitness;
        
        /*
        for(auto bestIndices : population.indicesElite){
            std::cout << "indicesElite: " << bestIndices;
        }
        std::cout << std::endl;
        
        for(int i=0; i<population.individuals.size(); i++){
            auto& net = population.individuals[i];
            printLine(); 
            std::cout << "Generation: " << g << " Network: " << i << " Fit: " << net.fitness << std::endl;
            printLine(); 
            std::cout << "type: " << net.startNode.type << " id: " << net.startNode.id << " edges: " << net.startNode.edges[0] << std::endl;
            for(const auto& n : net.innerNodes){
                std::cout << "type: " << n.type << " id: " << n.id << " F: " << n.f << " ";
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
        */
        bestFitnessPerGeneration.push_back(population.bestFit);
    }
    auto& net = population.individuals[population.individuals.size()-1];
    printLine(); 
    std::cout << "Best Network: " << " Fit: " << net.fitness << std::endl;
    printLine(); 
    printLine(); 
    std::cout << "type: " << net.startNode.type << " id: " << net.startNode.id << " edges: " << net.startNode.edges[0] << std::endl;
    for(const auto& n : net.innerNodes){
        std::cout << "type: " << n.type << " id: " << n.id << " F: " << n.f << " ";
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
    printVec(bestFitnessPerGeneration,"Best Fitness Values");
    printVec(data.XIndices, "Xindices");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "done in:" << duration.count() << "sek. \n"; 
    return 0;
}
