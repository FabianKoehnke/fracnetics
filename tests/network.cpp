#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>
#include "../include/Network.hpp"

class NetworkRemapTest : public ::testing::Test {
protected:
    std::shared_ptr<std::mt19937_64> generator;
    
    void SetUp() override {
        generator = std::make_shared<std::mt19937_64>(42); // Fixed seed for reproducibility
    }
};

TEST_F(NetworkRemapTest, RemapNodeIdsAndEdges_BasicRemapping) {
    // Create a simple network with 2 judgment nodes and 2 processing nodes
    Network net(generator, 2, 3, 2, 2, false);
    
    // Setup: Manually set some node IDs and edges for testing
    net.innerNodes[0].id = 0;
    net.innerNodes[1].id = 1;
    net.innerNodes[2].id = 2;
    net.innerNodes[3].id = 3;
    
    // Set some edges that will be remapped
    net.innerNodes[0].edges = {1, 2};
    net.innerNodes[1].edges = {2, 3};
    net.innerNodes[2].edges = {0};
    net.innerNodes[3].edges = {1};
    
    // Create a mapping: 0->10, 1->11, 2->12, 3->13
    std::unordered_map<int, int> map = {{0, 10}, {1, 11}, {2, 12}, {3, 13}};
    
    // Indices of nodes to remap (all nodes in this case)
    std::vector<int> nodeIndices = {0, 1, 2, 3};
    
    // Execute the remapping
    net.remapNodeIdsAndEdges(map, nodeIndices);
    
    // Verify node IDs were remapped
    EXPECT_EQ(net.innerNodes[0].id, 10);
    EXPECT_EQ(net.innerNodes[1].id, 11);
    EXPECT_EQ(net.innerNodes[2].id, 12);
    EXPECT_EQ(net.innerNodes[3].id, 13);
    
    // Verify edges were remapped
    EXPECT_EQ(net.innerNodes[0].edges[0], 11);
    EXPECT_EQ(net.innerNodes[0].edges[1], 12);
    EXPECT_EQ(net.innerNodes[1].edges[0], 12);
    EXPECT_EQ(net.innerNodes[1].edges[1], 13);
    EXPECT_EQ(net.innerNodes[2].edges[0], 10);
    EXPECT_EQ(net.innerNodes[3].edges[0], 11);
}

TEST_F(NetworkRemapTest, RemapNodeIdsAndEdges_PartialRemapping) {
    // Create a network
    Network net(generator, 2, 3, 2, 2, false);
    
    // Setup nodes
    net.innerNodes[0].id = 0;
    net.innerNodes[1].id = 1;
    net.innerNodes[2].id = 2;
    net.innerNodes[3].id = 3;
    
    net.innerNodes[0].edges = {1, 2};
    net.innerNodes[1].edges = {0, 3};
    
    // Only remap nodes 0 and 2
    std::unordered_map<int, int> map = {{0, 100}, {2, 200}};
    std::vector<int> nodeIndices = {0, 1};
    
    net.remapNodeIdsAndEdges(map, nodeIndices);
    
    // Node 0 should be remapped
    EXPECT_EQ(net.innerNodes[0].id, 100);
    // Node 1 should be unchanged
    EXPECT_EQ(net.innerNodes[1].id, 1);
    // Node 2 and 3 should not be touched (not in nodeIndices)
    EXPECT_EQ(net.innerNodes[2].id, 2);
    EXPECT_EQ(net.innerNodes[3].id, 3);
    
    // Edges: node 0's edges should be updated where mapping exists
    EXPECT_EQ(net.innerNodes[0].edges[0], 1);   // 1 not in map, unchanged
    EXPECT_EQ(net.innerNodes[0].edges[1], 200); // 2->200
    
    // Node 1's edges should be updated
    EXPECT_EQ(net.innerNodes[1].edges[0], 100); // 0->100
    EXPECT_EQ(net.innerNodes[1].edges[1], 3);   // 3 not in map, unchanged
}

TEST_F(NetworkRemapTest, RemapNodeIdsAndEdges_EmptyMapping) {
    // Create a network
    Network net(generator, 2, 2, 1, 2, false);
    
    int originalId = net.innerNodes[0].id;
    std::vector<int> originalEdges = net.innerNodes[0].edges;
    
    // Empty mapping should leave everything unchanged
    std::unordered_map<int, int> map;
    std::vector<int> nodeIndices = {0, 1};
    
    net.remapNodeIdsAndEdges(map, nodeIndices);
    
    EXPECT_EQ(net.innerNodes[0].id, originalId);
    EXPECT_EQ(net.innerNodes[0].edges, originalEdges);
}

