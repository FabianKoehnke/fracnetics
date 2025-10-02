#include "../include/Data.hpp"
#include "../include/Population.hpp"
#include "../include/PrintHelper.hpp"
#include <chrono>
#include <vector>

float accuracy(std::vector<int> v1, std::vector<int> v2){
    float acc = 0;
    for(int i=0; i<v1.size(); i++){
        if(v1[i]==v2[i]){
            acc++;
        }
    }
    acc /= v1.size();
    return acc;
}

int main(){
    /**
     * Parameter
     */
    float probEdgeMutationStartNode = 0.03;
    float probEdgeMutationInnerNodes = 0.03;
    float probBoundaryMutation = 0.1;
    float sigmaBoundaryMutationNormal = 0.01;
    std::string boundaryMutationType = "normal"; // uniform, networkSigma, normal, edgeSigma, edgeFractal
    bool fractalJudgment = false;
    float probCrossOver = 0.05;
    int generations = 200;
    int generationsNoImprovementLimit = 500;
    int nIndividuals = 3000;
    int tournamentSize = 2;
    int nElite = 1;
    int jn = 1;
    int jnf = 4;
    int pn = 2;
    int pnf = 3;
    int dMax = 10;
    int penalty = 2;
    int maxConsecutiveP = 2;
    int addDel = 1;
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
    std::vector<int> xIndices = {1,2,3,4};
    data.xySplit(5,xIndices);

    std::vector<int> yInt;
    for(float v : data.y){
        yInt.push_back(static_cast<int>(v));
    }

    std::cout << "X rows: " << data.X.size() << std::endl;
    std::cout << "X columns: " << data.X[0].size() << std::endl;
    printLine();
    data.minMaxFeatures(data.X); // calculate the min and max values of X (for node boundaries)
    // for cartpole just empty data needed
    printVec(data.minX, "minX");
    printVec(data.maxX, "miny");
    /**
     * Initializing the population
     */
    Population population(
            123, // seed 
            nIndividuals, // number of networks
            jn, // number of judgment nodes (jn)
            jnf, // number of jn functions 
            pn, // number of processing nodes (pn)
            pnf, // number of pn functions
            fractalJudgment
            ); 

    population.setAllNodeBoundaries(data.minX, data.maxX);
    printLine(); 
    std::cout << "start EA" << std::endl;
    std::vector<float> bestFitnessPerGeneration;
    int improvementCounter = 0;
    for(int g=0; g<generations; g++){
        //generator = std::make_shared<std::mt19937_64>(5494+g);
        population.accuracy(data.X,yInt,dMax,penalty);
        //population.callTraversePath(data.X, dMax);
        //for(auto& ind : population.individuals){
        //    ind.fitness = accuracy(yInt,ind.decisions);
        //}
        population.tournamentSelection(tournamentSize,nElite);
       
        population.crossover(probCrossOver);
        if(addDel == 1){
            population.callAddDelNodes(data.minX, data.maxX);
        }
        population.callEdgeMutation(probEdgeMutationInnerNodes, probEdgeMutationStartNode);

        std::cout << 
            "Geneation: " << g << 
            " BestFit: " << population.individuals[population.indicesElite[0]].fitness << 
            " MeanFitness: " << population.meanFitness << 
            " MinFitness: " << population.minFitness <<
            " NetworkSize Best Ind: " << population.individuals[population.indicesElite[0]].innerNodes.size() << std::endl;

       bestFitnessPerGeneration.push_back(population.bestFit);
       if(g > 1 && bestFitnessPerGeneration[g-1] == bestFitnessPerGeneration[g]){
           improvementCounter++;
           if(improvementCounter == generationsNoImprovementLimit){
               break;
            }
        } else {
            improvementCounter = 0;
        }
    }
    auto& net = population.individuals[population.individuals.size()-1];
    printLine(); 
    std::cout << "Best Network: " << " Fit: " << net.fitness << std::endl;
    printLine(); 
    printLine(); 
    std::cout << "type: " << net.startNode.type << " id: " << net.startNode.id << " edge: " << net.startNode.edges[0] << std::endl;
    int nodeCounter = 0;
    for(const auto& n : net.innerNodes){
        std::string usedNodeMarker;
        if(n.used==false){
            usedNodeMarker = "*";
        }else{
            usedNodeMarker = "";
        }
        nodeCounter ++;
        std::cout << usedNodeMarker << "type: " << n.type << " id: " << n.id << " F: " << n.f << " k: " << n.k_d.first << " d: " << n.k_d.second << " ";
        std::cout << "edges " << "(" << n.edges.size() << "): ";
        for(auto& ed : n.edges){
            std::cout << ed << " ";
        }

        std::cout << "boundaries" << "(" << n.boundaries.size() << "): ";
        for(auto& b: n.boundaries){
            std::cout << b << " ";
        }
        std::cout << "Frac Parameter: ";
        for(auto& p: n.productionRuleParameter){
            std::cout << p << " ";
        }
        std::cout << std::endl;
    }
    printLine();
    //printVec(bestFitnessPerGeneration,"Best Fitness Values");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "done in:" << duration.count() << "sek. \n"; 
    
    int sumTestFitness = 0;
    int tests = 10;
    printLine();
    std::cout << "Validation" << std::endl;
    return 0;
}
