#ifndef FRACTAL_HPP
#define FRACTAL_HPP
#include <vector>
#include <iostream>
#include <random>

/**
 * @file Fractal.hpp
 * @brief Utility functions for fractal-based edge pattern generation in GNP judgment nodes.
 * 
 * @details
 * This file provides utilities for creating hierarchical, self-similar decision
 * structures in judgment nodes. The fractal approach enables:
 * 
 * - **Hierarchical partitioning**: Feature space divided using L-system-inspired recursive patterns
 * - **Evolutionary optimization**: Production rule parameters can evolve to discover effective patterns
 * 
 * The fractal pattern generation follows the process:
 * 1. Determine k (branching factor) and d (depth) such that k^d equals number of edges
 * 2. Generate production rule parameters (random cut points in [0,1])
 * 3. Calculate relative lengths through recursive application of the production rule
 * 4. Map these lengths to actual feature space boundaries
 * 
 * This approach is inspired by L-systems and provides a biologically-motivated alternative
 * to uniform or arbitrary boundary placement.
 */

/**
 * @brief Calculates valid (k, d) combinations where k^d ≤ N for fractal edge structure.
 * 
 * @details
 * This function determines all valid fractal parameters for a judgment node with N possible
 * successor nodes. The fractal structure is defined by:
 * - **k**: Branching factor (number of subdivisions at each level)
 * - **d**: Depth of recursion (number of hierarchical levels)
 * 
 * **Algorithm**:
 * 1. Generates all (k, d) pairs where k^d ≤ N
 * 2. Enforces constraint: d ≥ 2 for N > 3 (ensures true fractal structure)
 * 3. Exception: d = 1 allowed only for N = 2 (binary decision case)
 * 4. Randomly selects one valid combination from all candidates
 * 
 * **Examples**:
 * - N = 8: possible combinations include (2,3), (2,2), (4,1) → returns one randomly
 * - N = 27: possible combinations include (3,3), (9,1), (3,2) → returns one randomly
 * - N = 2: only (2,1) is valid (binary split, no true fractal)
 * 
 * The random selection ensures diversity in fractal structures across different judgment
 * nodes, even when they have the same number of edges.
 * 
 * **Constraints**:
 * - k ≥ 2 (at least binary branching)
 * - d ≥ 2 for N > 3 (hierarchical structure required)
 * - k^d ≤ N (must not exceed available successor nodes)
 * 
 * @param N Number of available successor nodes (must be ≥ 2)
 * @param generator Shared pointer to random number generator for selecting from valid combinations
 * @return std::pair<int,int> Randomly selected (k, d) combination where k is branching factor and d is depth
 * 
 * @note Multiple calls with same N can return different (k, d) combinations
 */
inline std::pair<int,int> random_k_d_combination(int N, std::shared_ptr<std::mt19937_64> generator){
    int k=2;
    int start;
    if(N <= 3){start = 1;}else{start = 2;}
    std::vector<std::pair<int, int>> results;
    while(pow(k,1)<=N){
        int d=start;
        while (pow(k,d)<=N) { 
            results.push_back({k,d});
            d++;
        }
    k++;
    }
    std::uniform_int_distribution<int> distributionUniform(0,results.size()-1);
    int randomIndex = distributionUniform(*generator);
    return results[randomIndex];
}

/**
 * @brief Generates N random cut points in the interval (0, 1) for production rule parameterization.
 * 
 * @details
 * This function creates a set of random parameters that define where the feature space
 * should be subdivided. The parameters represent relative positions
 * for cuts in a normalized interval.
 * 
 * **Process**:
 * 1. Initializes vector with 0 (left boundary)
 * 2. Draws N random values from uniform distribution over (0, 1)
 * 3. Appends 1 (right boundary)
 * 4. Returns vector: [0, v₁, v₂, ..., vₙ, 1]
 * 
 * **Important**: The returned values are NOT sorted. Sorting and distance calculation
 * is performed separately by sortAndDistance() to obtain actual subdivision ratios.
 * 
 * **Example** (N=2):
 * - Random draws: 0.3, 0.7
 * - Returned: [0, 0.3, 0.7, 1]
 * - After sortAndDistance(): [0.3, 0.4, 0.3] (three intervals of relative sizes)
 * 
 * @param N Number of random cut points to generate
 * @param generator Shared pointer to random number generator for uniform sampling
 * @return std::vector<float> Vector of size N+2 containing: [0, random₁, random₂, ..., randomₙ, 1]
 * 
 * @note Uses std::nextafter(0.0f, 1.0f) to exclude exact 0 and 1 from the random distribution
 * @see sortAndDistance() and fractalLengths() for converting random cuts to fractal interval lengths
 */
inline std::vector<float> randomParameterCuts(int N, std::shared_ptr<std::mt19937_64> generator){
    std::vector<float> productionRuleParameter {0};
    for(int i=0; i<N; i++){
        std::uniform_real_distribution<float> distributionUniform(std::nextafter(0.0f, 1.0f), 1);
        float v = distributionUniform(*generator);
        productionRuleParameter.push_back(v);
    }
    productionRuleParameter.push_back(1);
    return productionRuleParameter;
}

/**
 * @brief Sorts cut points and calculates the relative distances (interval lengths) between consecutive points.
 * 
 * @details
 * This function converts a set of random cut points into a normalized vector of relative
 * interval lengths that sum to 1.0. This is a crucial step in fractal pattern generation,
 * transforming arbitrary parameter positions into usable subdivision ratios.
 * 
 * **Algorithm**:
 * 1. Sorts the input values in ascending order
 * 2. Calculates distances between consecutive sorted values: dᵢ = value[i+1] - value[i]
 * 3. Removes the last element (which would be redundant)
 * 4. Returns vector of relative lengths
 * 
 * **Mathematical property**: Since input starts with 0 and ends with 1, the returned
 * distances always sum to exactly 1.0, making them valid relative lengths for partitioning.
 * 
 * **Example**:
 * - Input:  [0, 0.4, 0.1, 0.5, 1]
 * - Sorted: [0, 0.1, 0.4, 0.5, 1]
 * - Distances: [0.1, 0.3, 0.1, 0.5]
 * - Verification: 0.1 + 0.3 + 0.1 + 0.5 = 1.0
 * 
 * These relative lengths represent the production rule for fractal generation: when
 * applied recursively, they create a self-similar hierarchical structure.
 * 
 * @param value Vector of unsorted cut points (typically from randomParameterCuts(), must start with 0 and end with 1)
 * @return std::vector<float> Sorted relative distances (interval lengths) between consecutive points, summing to 1.0
 * 
 * @note First element should be 0, last element should be 1 (for valid normalization)
 * @note Returned vector has size = input.size() - 1
 * @see randomParameterCuts() for generating the input parameters
 * @see fractalLengths() for applying these lengths recursively
 */
inline std::vector<float> sortAndDistance(std::vector<float> value){
    std::sort(value.begin(), value.end());
    for(int i=0; i<value.size()-1; i++){
       value[i] = value[i+1] - value[i];
    }
    value.pop_back();
    return value;
}

/**
 * @brief Generates fractal pattern of interval lengths through recursive subdivision (L-system style).
 * 
 * @details
 * This function creates a hierarchical self-similar pattern by recursively applying a
 * production rule to subdivide intervals. It is inspired by L-systems (Lindenmayer systems).
 * 
 * **Algorithm**:
 * 1. Start with single unit interval: [1]
 * 2. For each recursion level (depth iterations):
 *    - For each current interval with length L:
 *      - Replace it with k sub-intervals
 *      - Each sub-interval has length L × parameter[i]
 * 3. Return the final set of interval lengths
 * 
 * **Example** (depth=2, parameter=[0.3, 0.7]):
 * - Level 0: [1]
 * - Level 1: [1×0.3, 1×0.7] = [0.3, 0.7]
 * - Level 2: [0.3×0.3, 0.3×0.7, 0.7×0.3, 0.7×0.7] = [0.09, 0.21, 0.21, 0.49]
 * - Result: Four intervals lenghts! with fractal distribution summing up to 1
 * 
 * **Properties**:
 * - Total intervals: k^depth (where k = parameter.size())
 * - Self-similar: pattern repeats at different scales
 * - Sum: If parameters sum to 1, all lengths sum to 1 (normalized)
 * 
 * **Biological inspiration**: Like plant branching patterns where main branches divide
 * into smaller branches following the same subdivision rule at each level.
 * 
 * @param depth Recursion depth (number of hierarchical levels, typically d from random_k_d_combination())
 * @param parameter Production rule ratios (relative lengths for subdivision, typically k values from sortAndDistance() afer randomParameterCuts())
 * @return std::vector<float> Fractal pattern of k^depth interval lengths (relative sizes, sum to 1.0 if parameters sum to 1.0)
 * 
 * @note These lengths are mapped to feature boundaries by Node::setEdgesBoundaries()
 * @see random_k_d_combination() for determining (k, depth) parameters
 * @see randomParameterCuts() for generatiing the random cuts
 * @see sortAndDistance() for generating the production rule parameters
 */
inline std::vector<float> fractalLengths(int depth, std::vector<float> parameter){
    std::vector<float> lengths {1};
    std::vector<float> temp {1};
    for(int i=0; i<depth; i++){
        temp.clear();
        for(auto len : lengths){
            for(auto para : parameter){
                temp.push_back(len * para);
            }
        }
    lengths = temp;
    }
    return lengths;
}

#endif
