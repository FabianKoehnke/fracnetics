#include "../include/Population.hpp"
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>
#include <set>

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
    // Path order lives in Node::lastVisitStep (timestamp of the last visit), NOT in
    // Node::traverseCounter (visit frequency) -- see Node.hpp. The start node must also be
    // used == true, otherwise findSuccessorNodes() takes the early return for unused nodes.
    // nSelectedNodes is set explicitly because the default -1 draws a random number and
    // would make the test flaky.
    Network ind1 = populationProcessingLoop.individuals[0];
    for(auto& node : ind1.innerNodes){ node.used = true; }
    ind1.innerNodes[0].lastVisitStep = 1;
    ind1.innerNodes[3].lastVisitStep = 2;
    ind1.innerNodes[6].lastVisitStep = 3;
    int nAllNodes = static_cast<int>(ind1.innerNodes.size());
    std::vector<int> succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 0, nAllNodes);
    std::vector<int> succesorsTrue = {0,3,6};
    EXPECT_EQ(succesorsFound, succesorsTrue);
    succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 3, nAllNodes);
    succesorsTrue = {3,6};
    EXPECT_EQ(succesorsFound, succesorsTrue);
    succesorsFound = populationProcessingLoop.findSuccessorNodes(ind1, 6, nAllNodes);
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



// seedSpecialist: host is much weaker than the donor on one seed -> the donor's
// visited sub-graph for that seed must be appended (additively) to the host,
// protected via generationReceived, tagged with a shared transplantBlockID and
// exactly one isBlockEntry node (the donor's own traversal start node).
TEST(PopulationTest, SeedSpecialistTransplantsDeficitSeedSubgraphAdditively) {
    Population pop(123, 2, 0, 0, 8, 2, false);

    auto& host  = pop.individuals[0];
    auto& donor = pop.individuals[1];

    // Host is weak on seed 0, donor is strong on seed 0 -- donor should be picked.
    host.fitnessValues  = {-100.0f, 50.0f};
    donor.fitnessValues = { 200.0f, 40.0f}; // only better on seed 0 -> seed 1 must be ignored

    donor.visitedNodesPerSeed = {
        {0, 2, 5}, // seed 0: donor's active sub-graph (must include its own start node)
        {0, 1, 3}  // seed 1: irrelevant, host is not deficient here
    };
    // donor's traversal always starts at the same node -- make node 0 the entry.
    donor.startNode.edges[0] = 0;

    size_t hostSizeBefore = host.innerNodes.size();
    std::vector<int> genMaterialDonorBefore = storeGenMaterial(donor);

    pop.crossover(1.0, "seedSpecialist", /*currentGeneration=*/7, /*crossoverProtection=*/3);

    // Donor must stay completely unmodified (purely additive transplant).
    EXPECT_EQ(genMaterialDonorBefore, storeGenMaterial(donor));

    // Host must have grown by exactly the union of seed-0's visited nodes (3 nodes).
    ASSERT_EQ(host.innerNodes.size(), hostSizeBefore + 3);

    int blockID = -1;
    int nEntryNodes = 0;
    for (size_t i = hostSizeBefore; i < host.innerNodes.size(); i++) {
        auto& node = host.innerNodes[i];
        EXPECT_EQ(node.id, (unsigned int)i);              // id kept consistent with position
        EXPECT_EQ(node.generationReceived, 7);             // protected like other crossover types
        EXPECT_NE(node.transplantBlockID, -1);
        if (blockID == -1) blockID = node.transplantBlockID;
        EXPECT_EQ(node.transplantBlockID, blockID);        // all appended nodes share one block id
        if (node.isBlockEntry) nEntryNodes++;
        EXPECT_FALSE(node.used);                           // dormant until reached via mutation
        for (int edge : node.edges) {                      // edges must stay within the valid host range
            EXPECT_GE(edge, 0);
            EXPECT_LT(edge, (int)host.innerNodes.size());
        }
    }
    EXPECT_EQ(nEntryNodes, 1); // exactly one designated entry node for the whole block

    // Host's original nodes/edges must be completely untouched (purely additive).
    for (size_t i = 0; i < hostSizeBefore; i++) {
        EXPECT_EQ(host.innerNodes[i].transplantBlockID, -1);
    }
}

// If no individual in the population improves on any of the host's seeds
// (no positive fitness difference anywhere), seedSpecialist must be a no-op.
TEST(PopulationTest, SeedSpecialistNoOpWhenNoDonorImproves) {
    Population pop(123, 2, 0, 0, 8, 2, false);

    auto& host  = pop.individuals[0];
    auto& donor = pop.individuals[1];

    host.fitnessValues  = {100.0f, 100.0f};
    donor.fitnessValues = { 10.0f,  10.0f}; // strictly worse on both seeds
    donor.visitedNodesPerSeed = {{0,1}, {0,2}};

    std::vector<int> genMaterialBefore = storeGenMaterial(host);
    pop.crossover(1.0, "seedSpecialist", 0, 3);
    EXPECT_EQ(genMaterialBefore, storeGenMaterial(host));
}


TEST(PopulationTest, UpdateAdaptiveKIncreasesOnHighSuccessRate) {
    Population pop(1, 6, 4, 2, 4, 2, false);
    pop.indicesElite = {}; // treat all as non-elite for this test
    for (auto& ind : pop.individuals) ind.fitness = 100.0f;

    float initialK = pop.adaptiveK_;
    // "always succeeds" criterion -> success rate 1.0 > target 0.2 -> k should increase
    auto alwaysSucceeds = [](const Network&){ return true; };
    for (int g = 0; g < 12; g++) {
        pop.updateAdaptiveK(alwaysSucceeds, /*useRechenberg=*/true, /*invert=*/false, /*windowSize=*/10);
    }
    EXPECT_GT(pop.adaptiveK_, initialK);
    EXPECT_LE(pop.adaptiveK_, 5.0f); // must respect kMax clamp
}

TEST(PopulationTest, UpdateAdaptiveKDecreasesOnLowSuccessRate) {
    Population pop(1, 6, 4, 2, 4, 2, false);
    pop.indicesElite = {};

    float initialK = pop.adaptiveK_;
    auto neverSucceeds = [](const Network&){ return false; };
    for (int g = 0; g < 12; g++) {
        pop.updateAdaptiveK(neverSucceeds, /*useRechenberg=*/true, /*invert=*/false, /*windowSize=*/10);
    }
    EXPECT_LT(pop.adaptiveK_, initialK);
    EXPECT_GE(pop.adaptiveK_, 0.1f); // must respect kMin clamp
}

TEST(PopulationTest, UpdateAdaptiveKNoOpWhenRechenbergDisabled) {
    Population pop(1, 6, 4, 2, 4, 2, false);
    pop.indicesElite = {};
    pop.adaptiveK_ = 1.0f;

    auto neverSucceeds = [](const Network&){ return false; };
    for (int g = 0; g < 12; g++) {
        pop.updateAdaptiveK(neverSucceeds, /*useRechenberg=*/false, /*invert=*/false, /*windowSize=*/10);
    }
    EXPECT_FLOAT_EQ(pop.adaptiveK_, 1.0f); // k must stay untouched
    EXPECT_EQ(pop.successRateHistory_.size(), 10u); // history is still tracked (capped at windowSize)
}

TEST(PopulationTest, UpdateAdaptiveKInvertDecreasesOnHighSuccessRate) {
    Population pop(1, 6, 4, 2, 4, 2, false);
    pop.indicesElite = {};

    float initialK = pop.adaptiveK_;
    // "always succeeds" -> success rate 1.0 > target -> with invert=true k should DECREASE
    auto alwaysSucceeds = [](const Network&){ return true; };
    for (int g = 0; g < 12; g++) {
        pop.updateAdaptiveK(alwaysSucceeds, /*useRechenberg=*/true, /*invert=*/true, /*windowSize=*/10,
                            /*targetSuccessRate=*/0.9f);
    }
    EXPECT_LT(pop.adaptiveK_, initialK);
    EXPECT_GE(pop.adaptiveK_, 0.1f); // must respect kMin clamp
}

TEST(PopulationTest, UpdateAdaptiveKInvertIncreasesOnLowSuccessRate) {
    Population pop(1, 6, 4, 2, 4, 2, false);
    pop.indicesElite = {};

    float initialK = pop.adaptiveK_;
    // "never succeeds" -> success rate 0.0 < target -> with invert=true k should INCREASE
    auto neverSucceeds = [](const Network&){ return false; };
    for (int g = 0; g < 12; g++) {
        pop.updateAdaptiveK(neverSucceeds, /*useRechenberg=*/true, /*invert=*/true, /*windowSize=*/10,
                            /*targetSuccessRate=*/0.9f);
    }
    EXPECT_GT(pop.adaptiveK_, initialK);
    EXPECT_LE(pop.adaptiveK_, 5.0f); // must respect kMax clamp
}

TEST(PopulationTest, SeedRobustLineageSuccessUsesLineageBaseline) {
    Population pop(1, 1, 4, 2, 4, 2, false);
    Network& ind = pop.individuals[0];
    ind.lineageN = 0;
    EXPECT_TRUE(Population::seedRobustLineageSuccess(ind)) << "no history yet -> neutral/optimistic default";

    // simulate a lineage with a stable, well-established history around fitness=500
    for (int i = 0; i < 20; i++) {
        ind.updateLineageStats(500.0f, false);
    }
    ind.fitness = 500.0f;
    EXPECT_TRUE(Population::seedRobustLineageSuccess(ind)) << "fitness matching long-run baseline -> success";

    ind.fitness = 10.0f; // a single, badly-seeded generation far below the robust baseline
    EXPECT_FALSE(Population::seedRobustLineageSuccess(ind)) << "fitness far below robust lineage baseline -> failure";
}

TEST(PopulationTest, SeedConsistentSuccessRequiresAllSeedsAboveBaseline) {
    Population pop(1, 1, 4, 2, 4, 2, false);
    Network& ind = pop.individuals[0];

    // no lineage history yet -> neutral default
    ind.lineageN = 0;
    ind.fitnessValues = {900.0f, 950.0f};
    EXPECT_TRUE(Population::seedConsistentSuccess(ind));

    // establish a stable long-run baseline around 500
    for (int i = 0; i < 30; i++) {
        ind.updateLineageStats(500.0f, false);
    }

    // ALL seeds consistently at/above baseline -> success (default requiredFraction=1.0)
    ind.fitnessValues = {520.0f, 610.0f, 505.0f};
    EXPECT_TRUE(Population::seedConsistentSuccess(ind));

    // high MEAN but one seed far below baseline -> failure with strict requiredFraction=1.0,
    // even though the mean fitness (~503) looks comparable to the baseline
    ind.fitnessValues = {1000.0f, 1000.0f, 10.0f};
    EXPECT_FALSE(Population::seedConsistentSuccess(ind)) << "one bad seed must break a strict (1.0) requiredFraction";

    // same batch, but relaxed requiredFraction=0.5 ("most" seeds) -> success (2/3 >= baseline)
    EXPECT_TRUE(Population::seedConsistentSuccess(ind, 0.5f));

    // empty fitnessValues -> neutral default (not enough data)
    ind.fitnessValues = {};
    EXPECT_TRUE(Population::seedConsistentSuccess(ind));
}

TEST(PopulationTest, BatchThresholdSuccessIgnoresLineageAndUsesAbsoluteThreshold) {
    Population pop(1, 1, 4, 2, 4, 2, false);
    Network& ind = pop.individuals[0];
    ind.lineageN = 0; // lineage tracking disabled scenario (useLineageFitness=false) -> stays 0

    // no fitnessValues yet -> neutral default
    ind.fitnessValues = {};
    EXPECT_TRUE(Population::batchThresholdSuccess(ind));

    // all seeds near-perfect (>= default threshold 900) -> success, despite lineageN==0
    ind.fitnessValues = {950.0f, 900.0f, 1000.0f};
    EXPECT_TRUE(Population::batchThresholdSuccess(ind));

    // one seed below threshold -> failure with strict requiredFraction=1.0
    ind.fitnessValues = {1000.0f, 1000.0f, 850.0f};
    EXPECT_FALSE(Population::batchThresholdSuccess(ind));

    // same batch, relaxed requiredFraction=0.5 -> success (2/3 >= threshold)
    EXPECT_TRUE(Population::batchThresholdSuccess(ind, 900.0f, 0.5f));

    // custom (lower) threshold makes the same batch succeed at requiredFraction=1.0
    EXPECT_TRUE(Population::batchThresholdSuccess(ind, 800.0f, 1.0f));
}

TEST(PopulationTest, SeedConsistencyRatioIsScaleIndependentAndContinuous) {
    Population pop(1, 1, 4, 2, 4, 2, false);
    Network& ind = pop.individuals[0];

    // too few seeds -> neutral default
    ind.fitnessValues = {500.0f};
    EXPECT_FLOAT_EQ(Population::seedConsistencyRatio(ind), 1.0f);

    // perfectly consistent, but at a LOW absolute level (early training) -> ratio still 1.0
    ind.fitnessValues = {200.0f, 200.0f, 200.0f};
    EXPECT_FLOAT_EQ(Population::seedConsistencyRatio(ind), 1.0f);

    // perfectly consistent at a HIGH absolute level (late training) -> ratio still 1.0
    ind.fitnessValues = {1000.0f, 1000.0f, 1000.0f};
    EXPECT_FLOAT_EQ(Population::seedConsistencyRatio(ind), 1.0f);

    // one severe outlier seed -> low ratio, regardless of absolute scale
    ind.fitnessValues = {1000.0f, 1000.0f, 100.0f};
    EXPECT_NEAR(Population::seedConsistencyRatio(ind), 0.1f, 1e-5f);

    // negative worst-seed reward -> clamped to 0 (not negative)
    ind.fitnessValues = {500.0f, -50.0f};
    EXPECT_FLOAT_EQ(Population::seedConsistencyRatio(ind), 0.0f);

    // degenerate: max <= 0 -> neutral default
    ind.fitnessValues = {-10.0f, -20.0f};
    EXPECT_FLOAT_EQ(Population::seedConsistencyRatio(ind), 1.0f);
}




TEST(PopulationTest, UpdateAdaptiveKUsesContinuousScoreAveragedOverPopulationAndWindow) {
    Population pop(1, 4, 4, 2, 4, 2, false);
    pop.indicesElite = {};
    pop.individuals[0].fitnessValues = {1000.0f, 1000.0f, 1000.0f}; // consistency 1.0
    pop.individuals[1].fitnessValues = {1000.0f, 1000.0f, 1000.0f}; // consistency 1.0
    pop.individuals[2].fitnessValues = {1000.0f, 1000.0f, 0.0f};    // consistency 0.0
    pop.individuals[3].fitnessValues = {1000.0f, 1000.0f, 0.0f};    // consistency 0.0
    // population-average consistency this generation = 0.5

    pop.updateAdaptiveK(
        [](const Network& ind){ return Population::seedConsistencyRatio(ind); },
        /*useRechenberg=*/false, /*invert=*/false, /*windowSize=*/10);
    EXPECT_NEAR(pop.successRateHistory_.back(), 0.5f, 1e-5f)
        << "per-generation score must be the population-averaged continuous consistency ratio";
}

// ─── "seedSpecialistReplace" ────────────────────────────────────────────────────────
// Second seed-specialist variant: instead of appending the donor sub-graph as dormant, it
// replaces the host's DEFICIT-EXCLUSIVE sub-graph -- exactly the nodes the host traverses
// only on the seeds where the donor is better. Nodes the host also uses on its good seeds
// (the backbone) must be preserved.
TEST(PopulationTest, SeedSpecialistReplaceOverwritesOnlyDeficitExclusiveNodes) {
    Population pop(123, 2, 0, 0, 8, 2, false);
    pop.indicesElite = {1}; // keeps the donor from being modified as a host itself

    auto& host  = pop.individuals[0];
    auto& donor = pop.individuals[1];

    // Three seeds: the donor is better only on seed 0, the host on seeds 1 and 2.
    host.fitnessValues  = {  0.0f, 100.0f, 100.0f};
    donor.fitnessValues = {200.0f,  50.0f,  50.0f};

    // Host: nodes 0 and 1 are used everywhere (backbone), node 2 only on the deficit
    // seed 0 -> only node 2 may be replaced.
    host.visitedNodesPerSeed  = {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}};
    // Donor: on the deficit seed 0 it traverses nodes 5 and 6.
    donor.visitedNodesPerSeed = {{5, 6}, {7}, {7}};

    const size_t hostSizeBefore = host.innerNodes.size();
    const std::vector<int> genMaterialDonorBefore = storeGenMaterial(donor);
    const unsigned int donorNode5F = donor.innerNodes[5].f;
    const unsigned int donorNode6F = donor.innerNodes[6].f;
    std::vector<unsigned int> hostFBefore;
    for (auto& node : host.innerNodes) hostFBefore.push_back(node.f);

    pop.crossover(1.0, "seedSpecialistReplace", /*currentGeneration=*/5, /*crossoverProtection=*/3);

    // The donor only hands out copies and stays unchanged.
    EXPECT_EQ(genMaterialDonorBefore, storeGenMaterial(donor));

    // One host node is replaced, the donor sub-graph has two -> the network grows by exactly
    // the size difference (overflow handled as in "randomWidth").
    ASSERT_EQ(host.innerNodes.size(), hostSizeBefore + 1);

    // The deficit-exclusive node 2 now carries the donor's genetic material, the overflow
    // is appended at the end.
    EXPECT_EQ(host.innerNodes[2].f, donorNode5F);
    EXPECT_EQ(host.innerNodes.back().f, donorNode6F);
    EXPECT_EQ(host.innerNodes[2].generationReceived, 5);
    EXPECT_EQ(host.innerNodes.back().generationReceived, 5);

    // The backbone (0, 1) and the nodes used only on good seeds (3, 4) are kept.
    EXPECT_EQ(host.innerNodes[0].f, hostFBefore[0]);
    EXPECT_EQ(host.innerNodes[1].f, hostFBefore[1]);
    EXPECT_EQ(host.innerNodes[3].f, hostFBefore[3]);
    EXPECT_EQ(host.innerNodes[4].f, hostFBefore[4]);

    // Inserted actively instead of dormant: no block id, contiguous ids, valid edges.
    for (size_t i = 0; i < host.innerNodes.size(); i++) {
        EXPECT_EQ(host.innerNodes[i].transplantBlockID, -1);
        EXPECT_FALSE(host.innerNodes[i].isBlockEntry);
        EXPECT_EQ(host.innerNodes[i].id, (unsigned int)i);
        for (int edge : host.innerNodes[i].edges) {
            EXPECT_GE(edge, 0);
            EXPECT_LT(edge, (int)host.innerNodes.size());
        }
    }
}

// If the host uses the same nodes on its deficit seeds as on its good seeds, there is no
// seed-specific substructure to replace -- the host stays untouched (no fallback to
// replacing the backbone).
TEST(PopulationTest, SeedSpecialistReplaceNoOpWithoutDeficitExclusiveNodes) {
    Population pop(123, 2, 0, 0, 8, 2, false);
    pop.indicesElite = {1};

    auto& host  = pop.individuals[0];
    auto& donor = pop.individuals[1];

    host.fitnessValues  = {  0.0f, 100.0f};
    donor.fitnessValues = {200.0f,  50.0f};
    host.visitedNodesPerSeed  = {{0, 1}, {0, 1}}; // same nodes everywhere
    donor.visitedNodesPerSeed = {{5, 6}, {7}};

    const std::vector<int> genMaterialBefore = storeGenMaterial(host);
    pop.crossover(1.0, "seedSpecialistReplace", 5, 3);
    EXPECT_EQ(genMaterialBefore, storeGenMaterial(host));
}

// The replacing variant is destructive -- elite individuals must be skipped as hosts
// (unlike the additive variant, which also supplies elites).
TEST(PopulationTest, SeedSpecialistReplaceSkipsEliteHosts) {
    Population pop(123, 2, 0, 0, 8, 2, false);
    pop.indicesElite = {0}; // this time the HOST is elite

    auto& host  = pop.individuals[0];
    auto& donor = pop.individuals[1];

    host.fitnessValues  = {  0.0f, 100.0f};
    donor.fitnessValues = {200.0f,  50.0f};
    host.visitedNodesPerSeed  = {{0, 1, 2}, {0, 1, 3}};
    donor.visitedNodesPerSeed = {{5, 6}, {7}};

    const std::vector<int> genMaterialBefore = storeGenMaterial(host);
    pop.crossover(1.0, "seedSpecialistReplace", 5, 3);
    EXPECT_EQ(genMaterialBefore, storeGenMaterial(host));

    // Cross-check: the additive variant does supply the same elite.
    pop.crossover(1.0, "seedSpecialistAppend", 5, 3);
    EXPECT_GT(host.innerNodes.size(), 8u);
}



