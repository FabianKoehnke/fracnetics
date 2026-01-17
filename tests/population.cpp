#include <gtest/gtest.h>
#include "../include/Population.hpp"
#include "../include/Network.hpp"

class AddOverhangNodesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize population with basic parameters
        population = std::make_unique<Population>(
            42,    // seed
            10,    // ni (number of individuals)
            5,     // jn (judgment nodes)
            3,     // jnf (judgment node functions)
            5,     // pn (processing nodes)
            2,     // pnf (processing node functions)
            false  // fractalJudgment
        );
    }

    std::unique_ptr<Population> population;
};

TEST_F(AddOverhangNodesTest, TransfersOverhangNodesFromLargerToSmallerParent) {
    auto generator = std::make_shared<std::mt19937_64>(42);

    // Create two parent networks with different sizes
    Network parent1(generator, 5, 3, 5, 2, false);
    Network parent2(generator, 5, 3, 3, 2, false);
    
    size_t initialParent1Size = parent1.innerNodes.size();
    size_t initialParent2Size = parent2.innerNodes.size();
    
    // Create successor vectors simulating different subnetwork sizes
    std::vector<int> successor1 = {0, 1, 2, 3, 4};  // 5 nodes
    std::vector<int> successor2 = {0, 1, 2};        // 3 nodes
    
    // Call addOverhangNodes
    population->addOverhangNodes(successor1, successor2, parent1, parent2);
    
    // Verify parent2 received the overhang nodes (2 nodes: indices 3, 4)
    int expectedOverhang = successor1.size() - successor2.size();
    EXPECT_EQ(parent2.innerNodes.size(), initialParent2Size + expectedOverhang);
    
    // Parent1 size should remain the same (nodes are copied, not removed)
    EXPECT_EQ(parent1.innerNodes.size(), initialParent1Size);
}

TEST_F(AddOverhangNodesTest, NoTransferWhenEqualSizes) {
    auto generator = std::make_shared<std::mt19937_64>(42);
    // Create two parent networks with equal sizes
    Network parent1(generator, 5, 3, 5, 2, false);
    Network parent2(generator, 5, 3, 5, 2, false);
    
    size_t initialParent2Size = parent2.innerNodes.size();
    
    // Create successor vectors with equal sizes
    std::vector<int> successor1 = {0, 1, 2};
    std::vector<int> successor2 = {0, 1, 2};
    
    // Call addOverhangNodes
    population->addOverhangNodes(successor1, successor2, parent1, parent2);
    
    // Verify no nodes were added (overhang = 0)
    EXPECT_EQ(parent2.innerNodes.size(), initialParent2Size);
}

TEST_F(AddOverhangNodesTest, TransfersCorrectNumberOfNodes) {
    auto generator = std::make_shared<std::mt19937_64>(42);
    Network parent1(generator, 5, 3, 10, 2, false);
    Network parent2(generator, 5, 3, 2, 2, false);
    
    size_t initialParent2Size = parent2.innerNodes.size();
    
    // Large difference in successor sizes
    std::vector<int> successor1 = {0, 1, 2, 3, 4, 5, 6};  // 7 nodes
    std::vector<int> successor2 = {0, 1};                 // 2 nodes
    
    population->addOverhangNodes(successor1, successor2, parent1, parent2);
    
    // Should add exactly 5 overhang nodes (7 - 2)
    int expectedOverhang = 5;
    EXPECT_EQ(parent2.innerNodes.size(), initialParent2Size + expectedOverhang);
}

TEST_F(AddOverhangNodesTest, HandlesEmptySmallerSuccessor) {
    auto generator = std::make_shared<std::mt19937_64>(42);
    Network parent1(generator, 5, 3, 5, 2, false);
    Network parent2(generator, 5, 3, 2, 2, false);
    
    size_t initialParent2Size = parent2.innerNodes.size();
    
    // Successor2 is empty, successor1 has nodes
    std::vector<int> successor1 = {0, 1, 2};
    std::vector<int> successor2 = {};
    
    population->addOverhangNodes(successor1, successor2, parent1, parent2);
    
    // Should add all nodes from successor1
    EXPECT_EQ(parent2.innerNodes.size(), initialParent2Size + successor1.size());
}

