#include "../include/Population.hpp"
#include <gtest/gtest.h>
#include <unordered_map>
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

    // Testing swap overhand 
    Population populationDiffNetworkSizes(
        123, // seed 
        2, // number of networks
        0, // number of judgment nodes (jn)
        0, // number of jn functions 
        10, // number of processing nodes (pn)
        2, // number of pn functions
        false
        ); 
    ind1 = populationDiffNetworkSizes.individuals[0];
    Network ind2 = populationDiffNetworkSizes.individuals[1];
    ind2.innerNodes.erase(ind2.innerNodes.begin());
    for(Node& node : ind2.innerNodes){node.id --;}

    // assert: changed networkes sizes  
    int size1 = ind1.innerNodes.size();
    int size2 = ind2.innerNodes.size();
    std::vector<int> successor1 = {1,2,3};
    std::vector<int> successor2 = {1,2};
    std::unordered_map<int, int> swapMap1 = {{2,1}, {3,2}};
    populationDiffNetworkSizes.swapOverhangNodes(successor1, successor2, ind1, ind2, swapMap1);
    EXPECT_EQ(size1, ind2.innerNodes.size());
    EXPECT_EQ(size2, ind1.innerNodes.size());

    // assert: consecutive node ids 
    std::vector<int> idsInd1True; for(int i=0; i<ind1.innerNodes.size(); i++){idsInd1True.push_back(i);}
    std::vector<int> idsInd1; for(int i=0; i<ind1.innerNodes.size(); i++){idsInd1.push_back(ind1.innerNodes[i].id);}
    EXPECT_EQ(idsInd1True, idsInd1);
    std::vector<int> idsInd2True; for(int i=0; i<ind2.innerNodes.size(); i++){idsInd2True.push_back(i);}
    std::vector<int> idsInd2; for(int i=0; i<ind2.innerNodes.size(); i++){idsInd2.push_back(ind2.innerNodes[i].id);}
    EXPECT_EQ(idsInd2True, idsInd2);
   
    // assert: new edge mapping
    






}
