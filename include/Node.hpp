#ifndef NODE_HPP
#define NODE_HPP
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
        /** @cond INTERNAL */
        unsigned int id; /**< Unique identifier of the node within the network */
        std::string type; /**< Node type: "S" (Start), "P" (Processing), or "J" (Judgment) */
        unsigned int f; /**< Node function: feature index for judgment nodes or output value for processing nodes */
        std::vector<int> edges; /**< Indices of successor nodes (outgoing edges) */
        std::vector<double> boundaries; /**< Decision boundaries for judgment nodes (divides feature space into intervals) */
        std::vector<float> productionRuleParameter; /**< Parameters for fractal-based edge generation (used when fractalJudgment is enabled) */
        std::pair<int, int> k_d; /**< Fractal parameters: k (base) and d (depth) for fractal edge structure */
        bool used = false; /**< Flag indicating whether this node was visited during network traversal */
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
         * This method creates the edge structure for different node types according to GNP rules:
         * 
         * **Judgment Nodes (type "J")**:
         * - Can have multiple outgoing edges (between 2 and nn-1), where nn is the number of 
         *   processing and judgment nodes of a Network
         * - Randomly shuffles candidates
         *  - If parameter k=0: randomly selects between 2 and nn-1 edges
         *  - If parameter k>0: selects exactly k edges
         * - Edges represent conditional branches based on feature value intervals
         * 
         * **Processing Nodes (type "P") and Start Nodes (type "S")**:
         * - Have exactly one outgoing edge
         * - Randomly selects a successor node (excluding self to prevent self-loops)
         * 
         * @note Self-loop are always prevented
         *
         * The edge selection ensures well-defined graph structure with no self-loops 
         * for different node types.
         *
         * @param type Node type string: "J" (Judgment), "P" (Processing), or "S" (Start)
         * @param nn Total number of nodes in the network (used to determine valid edge targets)
         * @param k Number of outgoing edges for judgment nodes (0 = random selection between 2 and nn-1, >0 = fixed number)
         * 
         */
        void setEdges(std::string type, int nn, int k=0){

            if (type == "J") {
                for(int i=0; i<nn; i++){
                    if(i != this->id){//prevents self-loop
                        edges.push_back(i);    
                    }
                } 
                std::uniform_int_distribution<int> distribution(2, nn-1);
                int randomInt = distribution(*generator);// sets a random number of outgoing edges
                std::shuffle(edges.begin(), edges.end(), *generator);
                if(k == 0){
                    edges = std::vector<int>(edges.begin(), edges.begin()+randomInt);
                }else{
                    edges = std::vector<int>(edges.begin(), edges.begin()+k);
                }
            } else if(type == "S" || type == "P"){
                bool noSelfLoop = false;
                while(noSelfLoop == false){// prevents self-loop
                    std::uniform_int_distribution<int> distribution(0, nn-1);
                    int randomInt = distribution(*generator);// set a random successor
                    if(randomInt != this->id){
                        edges = std::vector<int>{randomInt};
                        noSelfLoop = true;
                        }
                    }
                } else {
                edges = std::vector<int>{};
            }
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
         * one possible judgment outcome (edge selection). Therfore, the number of boundaries
         * is equal to the number of edges of a judgment node.
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
               if(lengths.size()==0){
                   span = (maxf - minf) / edges.size();
               }else {
                   span = (maxf - minf) * lengths[i];
               } 
               boundaries.push_back(sum);
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
         * 
         * @note No self-loops are introduced by the mutation and 
         * the edges vector maintains its original size
         * 
         */
        void edgeMutation(float propability, int nn){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(auto& edge : edges){
                bool result = distributionBernoulli(*generator);
                if(result){ 
                    edge = changeEdge(nn, edge);
                }
            }
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
         * @return New valid node index for the edge
         * 
         * @warning Requirements: nn > 2 (at least 3 nodes required to ensure a valid alternative exists
         * because of the constraints). Otherwise method could run indefinitely.
         */
        int changeEdge(int nn, int& edge){
            std::uniform_int_distribution<int> distributionUniform(0, nn-1);
            while(true){ 
                int randomInt = distributionUniform(*generator);// sets a random number of outgoing edges
                if(randomInt != this->id && randomInt != edge){// prevent self-loop and same edge
                    return randomInt;
                }
            }
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
         * 
         */
        void boundaryMutationUniform(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i=1; i<boundaries.size()-1; i++){ // only shift the inner boundaries
                bool result = distributionBernoulli(*generator);
                if(result){
                    std::uniform_real_distribution<float> distributionUniform(boundaries[i-1], boundaries[i+1]);
                    boundaries[i] = distributionUniform(*generator);

                }
            }
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
        void boundaryMutationFractal(float propability, const std::vector<float>& minf, const std::vector<float>& maxf){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i=1; i<productionRuleParameter.size()-1; i++){ // only shift the inner parameter: [0,shift,...,1]
                bool result = distributionBernoulli(*generator);
                if(result){
                    std::uniform_real_distribution<float> distributionUniform(productionRuleParameter[i-1], productionRuleParameter[i+1]);
                    productionRuleParameter[i] = distributionUniform(*generator); 
                    boundaries.clear();
                    std::vector<float> fractals = fractalLengths(k_d.second, sortAndDistance(productionRuleParameter));
                    setEdgesBoundaries(minf[f], maxf[f], fractals);
                }
                }
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
         * @param sigma standard deviation of the normal distribution (later scaled by mu)
         * 
         * @note The standard deviation `sigma` is scaled by the current boundary value (`mu`) 
         * to maintain a consistent relative mutation strength. This ensures that boundaries with larger magnitude
         * receive proportionally similar adjustments as smaller boundaries
         */
        void boundaryMutationNormal(float propability, float sigma){
            std::bernoulli_distribution distributionBernoulli(propability);
            for(int i = 1; i<boundaries.size()-1; i++){ // only shift the inner boundaries
                bool result = distributionBernoulli(*generator);
                if(result){
                    float mu = boundaries[i];
                    float scaledSigma = sigma * mu;
                    std::normal_distribution<float> distributionNormal(mu,scaledSigma);
                    float newBoundary = distributionNormal(*generator);
                    if(newBoundary > boundaries[i-1] && newBoundary < boundaries[i+1]){ // preventing overlapping boundaries
                        boundaries[i] = newBoundary; 
                    }
                }
            }
        }
        /** @} */

};
#endif
