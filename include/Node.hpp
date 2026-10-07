#ifndef NODE_HPP
#define NODE_HPP
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <random>
#include "Fractal.hpp"
#include <iostream>

/**
 * @class Node 
 *
 * @brief This class defines the node of the GNP graph.
 *
 * @details
 * A Node represents a fundamental building block in the Genetic Network Programming
 * (GNP) graph structure. Each node can be one of three types:
 * - **Start Node (S)** - The entry point for network execution
 * - **Processing Node (P)** - Executes actions and produces output decisions
 * - **Judgment Node (J)** - Evaluates conditional branches based on input features
 * 
 * Nodes are connected through directed edges that define the graph traversal through
 * the network. Judgment nodes use boundary-based decision rules to select which
 * outgoing edge to follow based on feature values.
 *
 * @nosubgrouping
 */
class Node {
    private:
        std::shared_ptr<std::mt19937_64> generator; /**< Shared pointer to random number generator for stochastic operations */
    
    public:
        /** Diagnostic: identity of the RNG this node shares with its network. */
        std::uintptr_t rngPointer() const {
            return reinterpret_cast<std::uintptr_t>(generator.get());
        }

        /** @cond INTERNAL */
        unsigned int id; /**< Unique identifier of the node within the network */
        std::string type; /**< Node type: "S" (Start), "P" (Processing), or "J" (Judgment) */
        unsigned int f; /**< Node function: feature index for judgment nodes or output value for processing nodes */
        std::vector<int> edges; /**< Indices of successor nodes (outgoing edges) */
        std::vector<double> boundaries; /**< Decision boundaries for judgment nodes (divides feature space into intervals) */
        std::vector<float> productionRuleParameter = {}; /**< Parameters for fractal-based edge generation (used when fractalJudgment is enabled) */
        std::pair<int, int> k_d; /**< Fractal parameters: k (base) and d (depth) for fractal edge structure */
        bool used = false; /**< Flag indicating whether this node was visited during network traversal */
        unsigned int traverseCounter = 0; /**< How OFTEN this node was entered since the last reset (a true visit
                                               count, incremented by one per entry). Accumulates over all seeds of a
                                               generation (only Network::initPathTraversal()/fitGymnasium(newRun=true)
                                               reset it), so it is a usage FREQUENCY -- used e.g. for the traversal-
                                               neighborhood filter in Population::findSuccessorNodes(). */
        unsigned int lastVisitStep = 0; /**< Value of the network-wide step counter (Network::traverseCounter) at the
                                             LAST entry of this node, i.e. a timestamp, not a count. Used to order
                                             nodes along the traversal (Population::findSuccessorNodes() selects the
                                             nodes entered AFTER a given start node). Kept separate from
                                             traverseCounter, which previously carried this timestamp and was
                                             therefore unusable as a frequency. */
        int generationReceived = -1;

        // ─── Grace period for recently active nodes (Network::ageUnusedNodes) ───
        bool everUsed = false;      /**< true once the node has been traversed at least once */
        int unusedSince = 0;        /**< Generations since the last traversal. 0 while the node is
                                         in use, 1 after one generation without it, and so on.
                                         addDelNodes() deletes at unusedSince >= nodeGracePeriod,
                                         so a value of n grants n-1 idle generations -- n=1 grants
                                         none and disables the protection entirely.
                                         Keeps a node that was active last generation from being
                                         deleted only because this generation's seed panel did not
                                         need it. Replaces the clusterLabels detour in addDelNodes();
                                         on that old path 74.9% of protected nodes had been active
                                         one generation earlier. */

        int frozen = 0; /**< If > 0, this node is immutable – excluded from all mutation operators */

        // ─── Seed-specialist transplant block (Population::crossover(type="seedSpecialist")) ───
        int transplantBlockID = -1; /**< -1 = not part of any transplanted sub-graph block. Otherwise identifies
                                          the transplant event this node belongs to (see Population::nextTransplantBlockID).
                                          Used together with isBlockEntry to restrict which nodes of the block may be
                                          targeted by edge mutations originating from OUTSIDE the block. */
        bool isBlockEntry = false; /**< Only meaningful if transplantBlockID != -1. True for the single node through
                                         which the transplanted block may be entered by edges mutated from nodes
                                         outside the block (see changeEdge()). Nodes belonging to the block itself are
                                         not restricted when mutating their own outgoing edges. */
        
        // ─── Experience-Weighted Judgment Node (EWJN) ─────────────────────────

        /**
         * @brief Stores accumulated experience for one outgoing edge of a JE node.
         *
         * @details
         * Uses Welford's online algorithm for numerically stable computation of
         * mean and variance for both the discounted return G and the feature value
         * obs[f] observed when this edge was chosen.
         * 
         * Option C: only the node's own feature f is tracked for similarity,
         * keeping the semantics of GNP (each judgment node "knows" one feature).
         */
        struct EdgeExperience {
            float meanObs    = 0.0f; ///< Welford mean  of obs[f] when this edge was chosen
            float m2Obs      = 0.0f; ///< Welford M2    of obs[f] (for variance)
            float meanReturn = 0.0f; ///< Welford mean  of discounted return G
            float m2Return   = 0.0f; ///< Welford M2    of G (for variance)
            int   n          = 0;    ///< Number of times this edge was chosen
            float maxG_seen  = -std::numeric_limits<float>::max();
            float minG_seen  =  std::numeric_limits<float>::max();
        };

        std::vector<EdgeExperience> edgeExperience; ///< one entry per outgoing edge; only used for type "JE"
        float gamma = 0.999f; ///< Discount factor for return G (evolvable parameter)
        float alpha = 0.1f;  ///< Mixing weight: P(use experience over reactive judge) in [0,1] (evolvable)


        /** @endcond */
        
        /** @name Constructor */
        /** @{ */
        /**
         * @brief Constructs a Node with specified parameters.
         * 
         * @details
         * Creates a new node in the GNP network with the given identifier, type, and function.
         * The node is initialized without edges or boundaries, which must be set separately
         * using setEdges() and setEdgesBoundaries() methods.
         * 
         * @param _generator Shared pointer to a Mersenne Twister random number generator (std::mt19937_64) for stochastic operations
         * @param _id Unique identifier for this node within the network. The id should be always the index of "innerNodes" in a Network.
         * @param _type Node type as string: 
         *  - "S" for Start Node, 
         *  - "P" for Processing Node, 
         *  - "J" for Judgment Node
         * @param _f Node function: for judgment nodes this is the feature index to evaluate; for processing nodes this is the output/action value
         */
        Node(
            std::shared_ptr<std::mt19937_64> _generator,
            unsigned int _id, 
            std::string _type,
            unsigned int _f
            ):
            generator(_generator),
            id(_id),
            type(_type),
            f(_f)
                
            {}
        /** @} */

        /** @name Member Functions */
        /** @{ */

        /**
         * @brief Initializes the outgoing edges of the node based on its type and network size.
         *
         * @details
         * This method creates the edge structure for different node types according to GNP rules.
         * 
         * When `candidates` is provided (non-empty), edges are sampled from this list only —
         * treating each candidate as one meta-node (e.g. cluster entry or unclustered node).
         * When `candidates` is empty, all nodes [0, nn-1] are valid targets (original behaviour).
         *
         * **Judgment Nodes (type "J")**:
         * - Draws between 2 and candidates.size()-1 (or nn-1) edges, or exactly k if k > 0.
         * - Samples without self-loop.
         *
         * **Processing / Start Nodes (type "P" / "S")**:
         * - Exactly one edge, sampled uniformly from valid targets, excluding self-loop.
         *
         * @param type       Node type: "J", "P", or "S".
         * @param nn         Total number of nodes (used when candidates is empty).
         * @param k          Fixed number of edges for J-nodes (0 = random 2..size-1).
         * @param candidates Optional pre-built list of valid target node IDs.
         *                   When non-empty, replaces the full [0, nn-1] candidate set.
         *                   Typically: {Entry(C_k) for each cluster} + {unclustered node IDs}.
         */
        void setEdges(std::string type, int nn, int k = 0,
                      const std::vector<int>& candidates = {})
        {
            edges.clear();

            // Valid targets: candidates if given, else [0, nn-1]; never self.
            std::vector<int> valid;

            if (!candidates.empty()) {
                valid.reserve(candidates.size());
                for (int c : candidates)
                    if (c != static_cast<int>(this->id))
                        valid.push_back(c);
            } else {
                valid.reserve(nn - 1);
                for (int i = 0; i < nn; i++)
                    if (i != static_cast<int>(this->id))
                        valid.push_back(i);
            }

            if (valid.empty()) return; // safety: nothing to connect to

            // Type-specific edge initialisation
            if (type == "J") {
                std::shuffle(valid.begin(), valid.end(), *generator);
                int maxEdges = static_cast<int>(valid.size());

                int nEdges;
                if (k > 0) {
                    nEdges = std::min(k, maxEdges);
                } else if (maxEdges < 2) {
                    nEdges = maxEdges; // take all if fewer than 2 available
                } else {
                    std::uniform_int_distribution<int> dist(2, maxEdges);
                    nEdges = dist(*generator);
                }
                edges = std::vector<int>(valid.begin(), valid.begin() + nEdges);

            } else if (type == "S" || type == "P") {
                std::uniform_int_distribution<int> dist(0, static_cast<int>(valid.size()) - 1);
                edges = std::vector<int>{ valid[dist(*generator)] };

            }
            // type "E" or unknown → edges stays empty
        } 

        /**
         * @brief Evaluates a feature value and determines which outgoing edge to follow.
         *
         * @details
         * This method implements the decision logic for judgment nodes by mapping a continuous
         * feature value to a discrete edge index. The feature space is divided into intervals
         * defined by the boundaries vector, and this function determines which interval contains
         * the given value.
         * 
         * **Algorithm**:
         * 1. **Boundary cases**:
         *    - If v ≤ first boundary: return edge 0 (leftmost interval)
         *    - If v ≥ last boundary: return last edge index (rightmost interval)
         * 
         * 2. **Binary search** (for inner values):
         *    - Efficiently locates the interval [boundaries[i], boundaries[i+1]) given v
         *    - Returns the index i such that boundaries[i] ≤ v < boundaries[i+1]
         * 
         * The boundaries divide the feature space into non-overlapping intervals, each
         * corresponding to one outgoing edge. This enables GNP judgment nodes to make
         * multi-way decisions based on continuous feature values.
         *
         * @param v Feature value for evaluation (continuous real number)
         * @return Index of the edge to follow (integer in range [0, edges.size()-1]), or -1 if an error occurs
         * 
         * @warning Boundaries must be in ascending order: boundaries[i] < boundaries[i+1]
         * @note This function uses binary search for efficient interval lookup
         * @note Return value -1 indicates an algorithmic error (should not occur with valid boundaries)
         */
        int judge(float v){
            
            if(v <= boundaries[0]){
                return 0;
            } else if(v >= boundaries.back()){
                return edges.size()-1;
            } else {// do binary search
                int minIndex = 0;
                int maxIndex = edges.size()-1;
                while(minIndex <= maxIndex){
                    int midIndex = minIndex + (maxIndex - minIndex) / 2;
                    if(v >= boundaries[midIndex] && v < boundaries[midIndex+1]){
                       return midIndex;
                    } else if(v < boundaries[midIndex]){
                        maxIndex = midIndex-1;
                    } else{
                        minIndex = midIndex+1;
                    }
                }
            }// end binary search  
            return -1; 
        }

        /** 
         * @brief Sets the decision boundaries that partition the feature space for judgment node.
         *
         * @details
         * This method creates the boundary values that divide the continuous feature space
         * into intervals, one for each outgoing edge. Each interval corresponds to
         * one possible judgment outcome (edge selection). Therefore, a judgment node with
         * n edges has n+1 boundaries.
         * 
         * **Two modes of operation**:
         * 
         * 1. **Uniform spacing (lengths vector empty)**:
         *    - Divides the range [minf, maxf] into edges.size() equal intervals
         *    - Each interval has width: (maxf - minf) / edges.size()
         * 
         * 2. **Custom spacing (lengths vector provided)**:
         *    - Uses relative lengths from the lengths vector
         *    - Each interval i has width: (maxf - minf) × lengths[i]
         *    - Enables fractal or non-uniform partitioning patterns
         *    - The values in member lengths should sum to 1.0 for proper coverage
         * 
         * The resulting boundaries vector contains edges.size()+1 values that define
         * the edges.size() intervals: [b[0], b[1]), [b[1], b[2]), ..., [b[n-1], b[n]]
         *
         * @param minf Minimum feature value (lower boundary of the feature space)
         * @param maxf Maximum feature value (upper boundary of the feature space)
         * @param lengths optional vector of relative interval lengths (each in range [0,1], should sum to 1.0)
         * 
         * @note If lengths is provided, it should have size edges.size() with values summing to 1.0
         * @note Uniform spacing is used when lengths is empty or has size 0
         */
        void setEdgesBoundaries(float minf, float maxf, std::vector<float> lengths = {}){ 
           float sum = minf;
           float span;
           for(int i = 0; i<edges.size()+1; i++){
               boundaries.push_back(sum);
               if(i == edges.size()) break;
               if(lengths.size()==0){
                   span = (maxf - minf) / edges.size();
               }else {
                   span = (maxf - minf) * lengths[i];
               } 
               sum += span;
           }
        }

        /**
         * @brief Stochastically mutates the outgoing edges of the node.
         *
         * @details
         * This method implements edge mutation, a key evolutionary operator in GNP that
         * modifies the network topology. For each outgoing edge, the function:
         * 
         * 1. Draws a random boolean from a Bernoulli distribution with given probability
         * 2. If true, replaces the current edge target with a new random valid node
         * 3. Uses changeEdge() to ensure the new target is valid (no self-loops, different from current)
         * 
         * Edge mutation allows the network structure to evolve by redirecting connections,
         * potentially discovering better execution paths through the graph. This operation
         * preserves the number of edges while changing their targets but multiple outgoing
         * edges can have the same node as an successor.
         * 
         * **Expected number of mutated edges**: probability × edges.size()
         *
         * @param propability Probability (in range [0.0, 1.0]) that each individual edge will be mutated
         * @param nn Total number of nodes in the network (used to determine valid mutation targets)
         * @param k If > 0, replaces propability with min(1, k/N), i.e. about k mutations per N eligible
         * edges. N == 0 yields probability 0; the cap at 1 keeps bernoulli_distribution valid (an invalid
         * probability is UB and can segfault).
         * @param N Number of eligible edges for the k/N rate (0 e.g. when justUsedNodes is set before
         * any node was marked used).
         *
         * @note No self-loops are introduced by the mutation and 
         * the edges vector maintains its original size
         * 
         * @param allNodes Optional pointer to the network's full innerNodes vector. When provided, restricts
         * mutation targets so that a node NOT belonging to a transplant block (see transplantBlockID) cannot be
         * redirected into an interior (non-entry) node of a transplant block -- only that block's designated
         * isBlockEntry node remains reachable this way (see Population::crossover(type="seedSpecialist")).
         * Nodes that ARE themselves part of a block are unrestricted (this only guards entry FROM outside the
         * block). Passing nullptr (default) preserves the original, unrestricted behavior.
         */
        bool edgeMutation(float propability, int nn, float k, int N, const std::vector<Node>* allNodes = nullptr){

            if (frozen > 0) return false;

            bool changed = false;
            const int N_MIN = 1;

            if(k > 0.0f){
                // Guard N == 0 and k/N > 1: invalid p for bernoulli_distribution is UB.
                propability = (N > 0) ? std::min(1.0f, k / static_cast<float>(N)) : 0.0f;
            }
            for(int i = 0; i < static_cast<int>(edges.size()); i++){

                float p = propability;

                if(type == "JE" && edgeExperience[i].n >= N_MIN){
                    float mean = edgeExperience[i].meanReturn;       // [0,1]
                    // float var  = (edgeExperience[i].n > 1) ? edgeExperience[i].m2Return / static_cast<float>(edgeExperience[i].n) : 1.0f;
                    // float std  = std::sqrt(var);
                    // float confidence = 1.0f / (1.0f + std);          // ∈ (0,1]
                    // float quality = mean * confidence;

                    float quality = edgeExperience[i].meanReturn;
                    p = propability * (1.0f - quality);
                    p = std::max(p, 0.001f);
                }

                std::bernoulli_distribution distributionBernoulli(p);
                if(distributionBernoulli(*generator)){
                    edges[i] = changeEdge(nn, edges[i], allNodes);
                    changed = true;
                    if(type == "JE"){
                        edgeExperience[i] = EdgeExperience{};
                    }
                }
            }
            return changed;
        }
        /**
         * @brief Selects a new random target node for an edge while avoiding self-loops and duplicates.
         *
         * @details
         * This method generates a new valid successor node for an edge by randomly sampling
         * from the set of all nodes until finding one that satisfies the constraints:
         * 
         * **Constraints**:
         * - Must not equal this->id (prevents self-loops)
         * - Must not equal the current edge value (ensures actual change)
         * - Must be in valid range [0, nn-1]
         * 
         * The method uses rejection sampling: it draws random integers until finding one
         * that meets the constraints. 
         *
         * @param nn Total number of nodes in the network (defines the valid range [0, nn-1])
         * @param edge Current edge value (by reference, though not modified by this function)
         * @param allNodes Optional pointer to the network's full innerNodes vector. If provided AND this node
         * itself is not part of a transplant block (transplantBlockID == -1), candidate targets that belong to
         * a transplant block but are not that block's isBlockEntry node are rejected -- so a "normal" node can
         * only reach into a transplanted sub-graph via its single designated entry node. Nodes that ARE
         * themselves part of a block remain fully unrestricted. Passing nullptr (default) preserves the
         * original, unrestricted behavior.
         * @return New valid node index for the edge
         * 
         * @warning Requirements: nn > 2 (at least 3 nodes required to ensure a valid alternative exists
         * because of the constraints). Otherwise method could run indefinitely.
         */
        int changeEdge(int nn, int& edge, const std::vector<Node>* allNodes = nullptr){
            std::uniform_int_distribution<int> distributionUniform(0, nn-1);
            const bool restrictBlockEntry = (allNodes != nullptr) && (this->transplantBlockID == -1);
            // Safety fallback for the (practically unreachable) case that rejection sampling
            // cannot find a valid candidate -- avoids an infinite loop.
            const int maxAttempts = 10000;
            for(int attempt = 0; attempt < maxAttempts; attempt++){
                int randomInt = distributionUniform(*generator);// random target node
                if(randomInt != this->id && randomInt != edge){// prevent self-loop and same edge
                    if(restrictBlockEntry && randomInt < static_cast<int>(allNodes->size())){
                        const Node& candidate = (*allNodes)[randomInt];
                        if(candidate.transplantBlockID != -1 && !candidate.isBlockEntry){
                            continue; // interior node of a transplant block -- only reachable via its entry node
                        }
                    }
                    return randomInt;
                }
            }
            // Fallback: return any valid target ignoring the block restriction rather than looping forever.
            int randomInt;
            do {
                randomInt = distributionUniform(*generator);
            } while(randomInt == this->id);
            return randomInt;
        }

         /**
         * @brief Mutates decision boundaries by shifting them within their adjacent intervals using uniform distribution.
         *
         * @details
         * This function implements boundary mutation for judgment nodes, allowing the decision
         * thresholds to evolve without changing the network topology. For each interior boundary:
         * 
         * **Mutation process**:
         * 1. Draws a Bernoulli random variable with given probability
         * 2. If true, samples a new boundary value uniformly from [boundaries[i-1], boundaries[i+1]]
         * 3. Replaces the old boundary with the new value
         * 
         * **Key properties**:
         * - Only inner boundaries are mutated (indices 1 to boundaries.size()-2)
         * - First and last boundaries remain fixed (preserve feature space range)
         * - Preserves monotonicity: boundaries[i-1] < boundaries[i] < boundaries[i+1]
         * 
         * This uniform sampling approach provides unbiased exploration of the boundary space,
         * allowing both small and large shifts with equal probability within the valid range.
         *
         * @param propability Probability (in range [0.0, 1.0]) that boundary will be mutated
         * @param k If > 0, replaces propability with min(1, k/N); N == 0 yields 0 (see edgeMutation()).
         * @param N Number of eligible boundaries for the k/N rate.
         *
         */
        bool boundaryMutationUniform(float propability, float k=0.0f, int N=1){

            if (frozen > 0) return false;

            bool changed = false;
            if(k > 0.0f){
                // Guard N == 0 and k/N > 1: invalid p for bernoulli_distribution is UB.
                propability = (N > 0) ? std::min(1.0f, k / static_cast<float>(N)) : 0.0f;
            }

            const int N_MIN = 1;

            for(int i=1; i<static_cast<int>(boundaries.size())-1; i++){

                float p = propability;

                if(type == "JE"){
                    float q0 = 0.0f;
                    if(edgeExperience[i-1].n >= N_MIN){
                        float var0        = (edgeExperience[i-1].n > 1) ? edgeExperience[i-1].m2Return / static_cast<float>(edgeExperience[i-1].n) : 1.0f;
                        float confidence0 = 1.0f / (1.0f + std::sqrt(var0));
                        //q0                = edgeExperience[i-1].meanReturn * confidence0;
                        q0                = edgeExperience[i-1].meanReturn;
                    }

                    float q1 = 0.0f;
                    if(edgeExperience[i].n >= N_MIN){
                        float var1        = (edgeExperience[i].n > 1) ? edgeExperience[i].m2Return / static_cast<float>(edgeExperience[i].n) : 1.0f;
                        float confidence1 = 1.0f / (1.0f + std::sqrt(var1));
                        //q1                = edgeExperience[i].meanReturn * confidence1;
                        q1                = edgeExperience[i].meanReturn;
                    }

                    float quality = std::max(q0, q1);

                    if(quality > 0.0f){
                        p = propability * (1.0f - quality);
                        p = std::max(p, 0.001f);
                    }
                }

                std::bernoulli_distribution distributionBernoulli(p);
                if(distributionBernoulli(*generator)){
                    std::uniform_real_distribution<float> distributionUniform(boundaries[i-1], boundaries[i+1]);
                    boundaries[i] = distributionUniform(*generator);
                    changed = true;
                    if(type == "JE")
                        edgeExperience[i-1] = EdgeExperience{};
                    if(type == "JE")
                        edgeExperience[i] = EdgeExperience{};
                }
            }
            return changed;
        }
        
        // TODO: mention Paper II

        /**
         * @brief Mutates boundaries by adjusting fractal production rule parameters and 
         * recalculating the fractal structure according to a L-System.
         *
         * @details
         * This specialized mutation operator is used for judgment nodes with fractal-based edge patterns.
         * Instead of directly mutating boundaries, it mutates the underlying production rule parameters
         * that generate the fractal structure (see fractalLengths()), then recomputes the boundaries accordingly.
         * 
         * **Mutation process**:
         * 1. For each inner production rule parameter (indices 1 to size-2):
         *    - Draws a Bernoulli random variable with given probability
         *    - If true, uniformly samples new value between adjacent parameters
         *    - Ensures parameters remain in [0, 1] and properly ordered
         * 
         * 2. After mutating any parameter:
         *    - Clears the current boundaries vector
         *    - Recomputes fractal lengths using sortAndDistance() and fractalLengths()
         *    - Regenerates boundaries using setEdgesBoundaries() with new fractal pattern
         * 
         * **Key properties**:
         * - Maintains the fractal structure defined by k_d parameters (k base, d depth)
         * - Production rule parameters remain ordered: 0 = p[0] < p[1] < ... < p[n-1] = 1
         * - Boundaries are recalculated to match the feature range [minf[f], maxf[f]]
         *
         * @param propability Probability (in range [0.0, 1.0]) that each production rule parameter will be mutated
         * @param minf Vector of minimum values for all features (indexed by this->f)
         * @param maxf Vector of maximum values for all features (indexed by this->f)
         * 
         * @note productionRuleParameter must be initialized with values in [0, 1]
         */
        bool boundaryMutationFractal(float propability, const std::vector<float>& minf, const std::vector<float>& maxf){

            if (frozen > 0) return false;

            bool changed = false;
            std::bernoulli_distribution distributionBernoulli(propability);
            if(productionRuleParameter.size() > 0){
                for(int i=1; i<static_cast<int>(productionRuleParameter.size())-1; i++){
                    bool result = distributionBernoulli(*generator);
                    if(result){
                        std::uniform_real_distribution<float> distributionUniform(productionRuleParameter[i-1], productionRuleParameter[i+1]);
                        productionRuleParameter[i] = distributionUniform(*generator);
                        changed = true;
                        boundaries.clear();
                        std::vector<float> fractals = fractalLengths(k_d.second, sortAndDistance(productionRuleParameter));
                        setEdgesBoundaries(minf[f], maxf[f], fractals);
                        if(type == "JE") edgeExperience.assign(edges.size(), EdgeExperience{});
                    }
                }
            }
            return changed;
        }
        /**
         * @brief Mutates decision boundaries by shifting them using a normal (Gaussian) distribution.
         *
         * @details
         * This method implements boundary mutation with a normal distribution centered at the
         * current boundary value. For each inner boundary:
         * 
         * **Mutation process**:
         * 1. Draws a Bernoulli random variable with given probability
         * 2. If true:
         *    - Centers a normal distribution at the current boundary (μ = boundaries[i])
         *    - Samples a new value from N(μ, σ²)
         *    - Accepts the new value only if it falls within [boundaries[i-1], boundaries[i+1]]
         *    - Rejects values that would cause boundary crossing or reordering
         * 
         * **Key properties**:
         * - Only inner boundaries are mutated (first and last remain fixed)
         * - Small shifts are more likely than large shifts (Gaussian distribution)
         * - Preserves strict ordering: boundaries[i-1] < boundaries[i] < boundaries[i+1]
         * 
         * **Comparison to uniform mutation**:
         * - Uniform: unbiased exploration, all valid positions equally likely
         * - Normal: biased toward small changes, enables fine-tuning
         * - Normal: better for local optimization, worse for global exploration
         * 
         * The sigma parameter controls mutation strength: small sigma → fine-tuning,
         * large sigma → larger jumps (but still biased toward center).
         *
         * @param propability Probability (in range [0.0, 1.0]) that each interior boundary will be mutated
         * @param sigma Relative standard deviation; multiplied by the smaller gap to the neighbouring
         * boundaries. JE nodes ignore it and use an experience-based sigma in [0.05, 0.5].
         * @param k optional adaptive parameter: if > 0, replaces propability with min(1, k/N); N == 0 yields 0 (see edgeMutation())
         * @param N optional total number of eligible boundaries (used for adaptive probability scaling)
         */
        bool boundaryMutationNormal(float propability, float sigma, float k=0.0f, int N=1){

            if (frozen > 0) return false;

            bool changed = false;
            if(k > 0.0f){
                // Guard N == 0 and k/N > 1: invalid p for bernoulli_distribution is UB.
                propability = (N > 0) ? std::min(1.0f, k / static_cast<float>(N)) : 0.0f;
            }

            const int N_MIN = 1;

            for(int i = 1; i<boundaries.size()-1; i++){

                float p = propability;
                float adaptiveSigma = sigma;

                if(type == "JE"){
                    float q0 = 0.0f;
                    if(edgeExperience[i-1].n >= N_MIN){
                        float var0        = (edgeExperience[i-1].n > 1) ? edgeExperience[i-1].m2Return / static_cast<float>(edgeExperience[i-1].n) : 1.0f;
                        float confidence0 = 1.0f / (1.0f + std::sqrt(var0));
                        //q0                = edgeExperience[i-1].meanReturn * confidence0;
                        q0                = edgeExperience[i-1].meanReturn;
                    }

                    float q1 = 0.0f;
                    if(edgeExperience[i].n >= N_MIN){
                        float var1        = (edgeExperience[i].n > 1) ? edgeExperience[i].m2Return / static_cast<float>(edgeExperience[i].n) : 1.0f;
                        float confidence1 = 1.0f / (1.0f + std::sqrt(var1));
                        //q1                = edgeExperience[i].meanReturn * confidence1;
                        q1                = edgeExperience[i].meanReturn;
                    }

                    float quality = std::max(q0, q1);

                    // if(quality > 0.0f){
                    //     p = propability * (1.0f - quality);
                    //     p = std::max(p, 0.001f);
                    // }
                    float sigmaMax = 0.5;
                    float sigmaMin = 0.05;
                    adaptiveSigma = sigmaMax * (1.0f - quality) + sigmaMin * quality;
                }

                std::bernoulli_distribution distributionBernoulli(p);
                if(distributionBernoulli(*generator)){
                    float mu = boundaries[i];

                    if(type != "JE"){
                        float lowerGap = boundaries[i] - boundaries[i-1];
                        float upperGap = boundaries[i+1] - boundaries[i];
                        adaptiveSigma = sigma * std::min(lowerGap, upperGap);
                    }
                    std::normal_distribution<float> distributionNormal(mu, adaptiveSigma);
                    float newBoundary = distributionNormal(*generator);
                    if(newBoundary > boundaries[i-1] && newBoundary < boundaries[i+1]){
                        boundaries[i] = newBoundary;
                        changed = true;
                        if(type == "JE")
                            edgeExperience[i-1] = EdgeExperience{};
                        if(type == "JE")
                            edgeExperience[i] = EdgeExperience{};
                    }
                }
            }
            return changed;
        }
        // ─── Experience-Weighted Judgment Node (EWJN): Methods ────────────────

        /**
         * @brief Initialises edgeExperience to match the current edges vector.
         *
         * @details
         * Must be called once after setEdges() for nodes of type "JE".
         * Safe to call multiple times; resets all accumulated experience.
         */
        void initEdgeExperience() {
            edgeExperience.assign(edges.size(), EdgeExperience{});
        }

        /**
         * @brief Updates the experience of one edge with a new (G, obs[f]) observation.
         *
         * @details
         * Applies Welford's online algorithm in-place. This is called during the
         * backward return pass at the end of each episode in fitGymnasium().
         *
         * @param edgeIdx  Index into edges[] (and edgeExperience[]) to update
         * @param G        Discounted return observed from time t onward
         * @param obsF     Value of obs[node.f] at the time this edge was chosen
         * @param G_min    Worst expected return (worstFitness); maps to 0 when G is min-max normalised
         * @param G_max    Best expected return; maps to 1. G_norm is clamped to [0, 1].
         */

        void updateEdgeExperience(int edgeIdx, float G, float obsF,
                          float G_min = -500.0f, float G_max = 250.0f) {
            EdgeExperience& e = edgeExperience[edgeIdx];

            // Min-max normalise: 0 = worst (worstFitness), 1 = best
            float G_norm = std::clamp((G - G_min) / (G_max - G_min), 0.0f, 1.0f);

            e.n++;

            // Welford for obsF
            float d1 = obsF - e.meanObs;
            e.meanObs  += d1 / e.n;
            e.m2Obs    += d1 * (obsF - e.meanObs);

            // Welford for G_norm
            float d2 = G_norm - e.meanReturn;
            e.meanReturn += d2 / e.n;
            e.m2Return   += d2 * (G_norm - e.meanReturn);
        }

        // void updateEdgeExperience(int edgeIdx, float G, float obsF, int remainingSteps) {
        //     EdgeExperience& e = edgeExperience[edgeIdx];
        //
        //     //G normieren: relativ zu maximal möglichem Return ab Zeitpunkt t
        //     float maxG       = (gamma < 1.0f)
        //                        ? (1.0f - std::pow(gamma, remainingSteps)) / (1.0f - gamma)
        //                        : static_cast<float>(remainingSteps);  // γ=1 → kein Discount
        //
        //     float G_norm     = (maxG > 0.0f) ? G / maxG : 0.0f;
        //
        //     // const float MAX_RETURN = 250.0f;
        //     // float G_norm = std::clamp(G / MAX_RETURN, 0.0f, 1.0f);
        //
        //     e.n++;
        //
        //     // Welford für obsF
        //     float d1 = obsF - e.meanObs;
        //     e.meanObs  += d1 / e.n;
        //     e.m2Obs    += d1 * (obsF - e.meanObs);
        //
        //     // Welford für G_norm statt G
        //     float d2 = G_norm - e.meanReturn;
        //     //float lr = 0.02f; // Lernrate
        //     //e.meanReturn = (1.0f - lr) * e.meanReturn + lr * G;
        //     e.meanReturn += d2 / e.n;
        //     e.m2Return   += d2 * (G_norm - e.meanReturn);
        // }
        //

        /**
         * @brief Experience-weighted edge selection (Option C: similarity on obs[f] only).
         *
         * @details
         * For the node's own feature f, computes a Gaussian similarity between the
         * current observation obsF and the mean observation stored per edge.
         * The variance used for the Gaussian kernel is estimated from the Welford M2.
         *
         * Score for edge i:  score_i = similarity_i * meanReturn_i
         *
         * With probability alpha the best-scored edge is selected (experience-based);
         * with probability (1 - alpha) the standard reactive judge() is used.
         * If no edge has been visited yet, falls back to judge() unconditionally.
         *
         * @param obsF  Current value of obs[node.f]
         * @return      Index into edges[] to follow
         */
        int judgeWithExperience(float obsF) {

            int reactiveEdge = judge(obsF);

            // Experience counts only after N_MIN visits
            const int N_MIN = 1; //       / static_cast<int>(edges.size());  
            bool anyExp = false;
            for (const auto& e : edgeExperience)
                if (e.n >= N_MIN) { anyExp = true; break; }
            if (!anyExp) return reactiveEdge;

            // ── Experience score per edge ───────────────────────────────────────────
            std::vector<float> eScores(edges.size(), 0.0f);
            float eMin =  std::numeric_limits<float>::max();
            float eMax = -std::numeric_limits<float>::max();

            for (int i = 0; i < static_cast<int>(edges.size()); i++) {
                const auto& e = edgeExperience[i];
                if (e.n > 0) {
                    float var  = (e.n > 1) ? e.m2Obs / static_cast<float>(e.n) : 1.0f;
                    if (var < 1e-6f) var = 1.0f;
                    float diff = obsF - e.meanObs;
                    float sim  = std::exp(-(diff * diff) / (2.0f * var));
                    eScores[i] = sim * e.meanReturn;
                    eMin = std::min(eMin, eScores[i]);
                    eMax = std::max(eMax, eScores[i]);
                }
            }

            // ── Normalise eScore to [0,1] ───────────────────────────────────────────
            float range = eMax - eMin;
            for (int i = 0; i < static_cast<int>(edges.size()); i++) {
                if (range > 1e-6f)
                    eScores[i] = (eScores[i] - eMin) / range;
                else
                    eScores[i] = 0.0f;
            }

            // ── Weighted score ──────────────────────────────────────────────────────
            int   bestEdge  = reactiveEdge;
            float bestScore = -std::numeric_limits<float>::max();

            for (int i = 0; i < static_cast<int>(edges.size()); i++) {
                float rScore = (i == reactiveEdge) ? 1.0f : 0.0f;
                float score  = alpha * eScores[i] + (1.0f - alpha) * rScore;
                if (score > bestScore) {
                    bestScore = score;
                    bestEdge  = i;
                }
            }
            return bestEdge;
        }

        /**
         * @brief Mutates the discount factor gamma uniformly in [0, 1].
         * @param probability Probability that gamma is resampled.
         */
        bool gammaMutation(float probability) {
            std::bernoulli_distribution dist(probability);
            if (dist(*generator)) {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                gamma = uniform(*generator);
                if (type == "JE") {
                    edgeExperience.assign(edges.size(), EdgeExperience{});
                }
                return true;
            }
            return false;
        }

        /**
         * @brief Mutates the mixing weight alpha uniformly in [0, 1].
         * @param probability Probability that alpha is resampled.
         */
        bool alphaMutation(float probability) {
            std::bernoulli_distribution dist(probability);
            if (dist(*generator)) {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                alpha = uniform(*generator);
                return true;
            }
            return false;
        }
        /** @} */

};
#endif
