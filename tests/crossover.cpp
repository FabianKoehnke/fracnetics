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

    // Testing initNodeSwapMap - function to initialize the node map 
    std::vector<int>subnodes1 = {1,2,3,10};
    std::vector<int>subnodes2 = {8,2,5};
    std::unordered_map<int, int> map = populationProcessingLoop.initNodeSwapMap(subnodes1, subnodes2, 10);
    std::vector<int>testMapKeys;
    std::vector<int>testMapValues;
    for(const auto& pair : map){
        testMapKeys.push_back(pair.first);
        testMapValues.push_back(pair.second);
    }
    std::sort(testMapKeys.begin(), testMapKeys.end());
    std::sort(subnodes1.begin(), subnodes1.end());
    subnodes2.push_back(10);
    std::sort(testMapValues.begin(), testMapValues.end());
    std::sort(subnodes2.begin(), subnodes2.end());
    // assert same kays and values in swap maps
    EXPECT_EQ(testMapKeys, subnodes1);
    EXPECT_EQ(testMapValues, subnodes2);

}

TEST(PopulationTest, BFSGraphDepthTraversal) {
    // Create a small population with processing nodes only (no judgment nodes)
    Population pop(
        42,  // seed
        2,   // number of networks
        0,   // number of judgment nodes
        0,   // number of jn functions
        10,  // number of processing nodes
        2,   // number of pn functions
        false
    );

    Network ind = pop.individuals[0];

    // Mark all nodes as used so findSuccessorNodes does not early-return
    for (auto& node : ind.innerNodes) {
        node.used = true;
    }

    // Wire a linear chain: 0 → 2 → 5 → 7
    // Clear existing edges first, then set desired edges
    for (auto& node : ind.innerNodes) {
        node.edges.clear();
    }
    ind.innerNodes[0].edges = {2};
    ind.innerNodes[2].edges = {5};
    ind.innerNodes[5].edges = {7};

    // graphDepth = 0: only the start node
    std::vector<int> result = pop.findSuccessorNodes(ind, 0, 10, false, 0);
    EXPECT_EQ(result, (std::vector<int>{0}));

    // graphDepth = 1: start node + direct neighbour
    result = pop.findSuccessorNodes(ind, 0, 10, false, 1);
    EXPECT_EQ(result, (std::vector<int>{0, 2}));

    // graphDepth = 2: two hops
    result = pop.findSuccessorNodes(ind, 0, 10, false, 2);
    EXPECT_EQ(result, (std::vector<int>{0, 2, 5}));

    // graphDepth = 3: full chain
    result = pop.findSuccessorNodes(ind, 0, 10, false, 3);
    EXPECT_EQ(result, (std::vector<int>{0, 2, 5, 7}));

    // Test cycle: node 7 → 0 (back edge); BFS should not infinite-loop
    ind.innerNodes[7].edges = {0};
    result = pop.findSuccessorNodes(ind, 0, 10, false, 3);
    EXPECT_EQ(result, (std::vector<int>{0, 2, 5, 7}));

    // Test nSelectedNodes cap: limit to 2 additional successors beyond the start node (total 3 nodes)
    ind.innerNodes[7].edges.clear();
    result = pop.findSuccessorNodes(ind, 0, 2, false, 3);
    EXPECT_EQ(result, (std::vector<int>{0, 2, 5}));
}
