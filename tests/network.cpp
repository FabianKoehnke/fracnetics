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

// changeEdge(): a node OUTSIDE any transplant block must never be redirected onto an
// interior (non-entry) node of a block -- only that block's designated entry node
// remains a valid mutation target. Nodes belonging to the block themselves stay
// unrestricted (self-mutation of the block's own edges is unaffected).
TEST_F(NetworkRemapTest, ChangeEdgeRespectsTransplantBlockEntryRestriction) {
    Network net(generator, 5, 3, 2, 2, false); // 5 judgment + 2 processing = 7 inner nodes

    // Mark nodes 2,3,4 as one transplant block; node 3 is the designated entry.
    net.innerNodes[2].transplantBlockID = 1;
    net.innerNodes[3].transplantBlockID = 1;
    net.innerNodes[3].isBlockEntry = true;
    net.innerNodes[4].transplantBlockID = 1;

    // Node 0 is a normal node (not part of the block) -- repeatedly mutate one of its
    // edges and verify it never lands on the block's interior nodes (2 or 4).
    for (int trial = 0; trial < 200; trial++) {
        int edge = 6; // some current target, irrelevant to the check itself
        int newTarget = net.innerNodes[0].changeEdge((int)net.innerNodes.size(), edge, &net.innerNodes);
        EXPECT_NE(newTarget, 2);
        EXPECT_NE(newTarget, 4);
        // reaching the entry node (3) itself is fine and expected to occur
    }

    // A node that IS part of the block (node 2) must remain unrestricted and CAN
    // target another interior node of the same block (e.g. node 4).
    bool reachedInteriorFromInsideBlock = false;
    for (int trial = 0; trial < 200; trial++) {
        int edge = 6;
        int newTarget = net.innerNodes[2].changeEdge((int)net.innerNodes.size(), edge, &net.innerNodes);
        if (newTarget == 4) { reachedInteriorFromInsideBlock = true; break; }
    }
    EXPECT_TRUE(reachedInteriorFromInsideBlock);
}


// addDelNodes() removes ALL deletable unused nodes per call (except the one left standing by
// the "more than one unused node" condition) -- not just every second one, as before the n--
// after erase() (the node shifted into place was skipped by the loop's n++). This is the
// counterweight to crossover(type="seedSpecialist") appending whole sub-graphs as blocks.
TEST_F(NetworkRemapTest, AddDelNodesRemovesAllUnusedNodesPerCall) {
    Network net(generator, 6, 3, 6, 4, false);

    const int originalSize = static_cast<int>(net.innerNodes.size());
    ASSERT_EQ(originalSize, 12);

    // Only nodes 0, 1 and 2 count as used -- the remaining 9 are deletable.
    for (auto& node : net.innerNodes) node.used = false;
    net.innerNodes[0].used = true;
    net.innerNodes[1].used = true;
    net.innerNodes[2].used = true;
    for (auto& node : net.innerNodes) node.generationReceived = -1; // no crossoverProtection

    std::vector<float> minF(4, 0.0f), maxF(4, 1.0f);
    std::vector<int> nFeatureValues;

    // addDelNodes() flips a coin per call between adding and deleting; with junk = 0 the add
    // branch is blocked while any unused node exists. Several calls ensure the delete branch
    // is drawn at least once.
    for (int call = 0; call < 20; call++) {
        net.addDelNodes(minF, maxF, 0.0f, nFeatureValues, /*currentGeneration=*/0, /*crossoverProtection=*/0);
    }

    int unusedLeft = 0;
    for (const auto& node : net.innerNodes) if (!node.used) unusedLeft++;

    // Exactly one unused node may remain (condition size - nUsed - 1 > size*junk), and all
    // three used nodes must be preserved.
    EXPECT_EQ(unusedLeft, 1);
    EXPECT_EQ(static_cast<int>(net.innerNodes.size()), 4);

    // After the deletions, ids must be contiguous again and all edges valid.
    for (int i = 0; i < static_cast<int>(net.innerNodes.size()); i++) {
        EXPECT_EQ(static_cast<int>(net.innerNodes[i].id), i);
        for (int edge : net.innerNodes[i].edges) {
            EXPECT_GE(edge, 0);
            EXPECT_LT(edge, static_cast<int>(net.innerNodes.size()));
        }
    }
    EXPECT_LT(net.startNode.edges[0], static_cast<int>(net.innerNodes.size()));
}
