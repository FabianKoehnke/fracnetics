#include "../include/Population.hpp"
#include <gtest/gtest.h>
#include <vector>

std::vector<int> storeGenMaterial(Network individual){
    std::vector<int> genmaterial;
    for(auto& node : individual.innerNodes){
        genmaterial.push_back(node.f);
        genmaterial.push_back(node.id);
        for(auto& edge : node.edges){
            genmaterial.push_back(edge);
        }
    }
    return genmaterial;
}

TEST(PopulationTest, BasicAssertions) {
    Population populationUniform(
        123, // seed 
        2, // number of networks
        5, // number of judgment nodes (jn)
        5, // number of jn functions 
        5, // number of processing nodes (pn)
        5, // number of pn functions
        false
        ); 

    /*
     * Testing uniform crossover
     */

    // test for differences in genmaterial after applying crossover with probability = 1
    std::vector<int> genMaterialBefore = storeGenMaterial(populationUniform.individuals[0]);
    populationUniform.crossover(1, "uniform");
    std::vector<int> genMaterialAfter = storeGenMaterial(populationUniform.individuals[0]);
    EXPECT_NE(genMaterialBefore, genMaterialAfter);

    /* 
    * Testing random width crossover
    */
    Population populationProcessingLoop(
        123, // seed 
        2, // number of networks
        0, // number of judgment nodes (jn)
        0, // number of jn functions 
        10, // number of processing nodes (pn)
        2, // number of pn functions
        false
        ); 
    
    // Testing successor nodes
    Network ind1 = populationProcessingLoop.individuals[0];
    ind1.innerNodes[0].traverseCounter = 1;
    ind1.innerNodes[3].traverseCounter = 2;
    ind1.innerNodes[6].traverseCounter = 3;
    std::vector<int> succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 0);
    std::vector<int> succesorsTrue = {0,3,6};
    EXPECT_EQ(succesorsFound, succesorsTrue);
    succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 3);
    succesorsTrue = {3,6};
    EXPECT_EQ(succesorsFound, succesorsTrue);
    succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 6);
    succesorsTrue = {6};
    EXPECT_EQ(succesorsFound, succesorsTrue);





}
