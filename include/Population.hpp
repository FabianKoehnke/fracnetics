#ifndef POPULATION_HPP
#define POPULATION_HPP
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>
#include <unordered_set>
#include <utility>
#include <cmath>
#include "Network.hpp"
#include "GymnasiumWrapper.hpp"

/**
 * @class Population 
 * @brief Manages the evolutionary population of GNP individuals (networks).
 *
 * @details
 * The Population class implements the evolutionary framework for Genetic Network Programming (GNP).
 * It provides all necessary operations for evolutionary optimization:
 * 
 * - **Population Management**: Initialization and maintenance of multiple network individuals
 * - **Fitness Evaluation**: Applying various fitness functions (accuracy, CartPole, Gymnasium environments)
 * - **Selection**: Tournament selection with elitism (to preserve best individuals)
 * - **Genetic Operators**: 
 *   - Edge mutation (topology changes)
 *   - Boundary mutation (parameter tuning)
 *   - Crossover (recombination of network structures)
 *   - Node addition/deletion (structural evolution)
 * 
 * The population evolves over generations through iterative applying fitness measurments,
 * selection, and mutation operators, which gradually improves the collective fitness and
 * discovering effective network structures for the target problem. A small tutorial can be found here:
 *
 * https://colab.research.google.com/github/FabianKoehnke/fracnetics/blob/main/notebooks/minExampleCartPole.ipynb
 *
 * @nosubgrouping
 */
class Population {
    private:
        std::shared_ptr<std::mt19937_64> generator; /**< Shared pointer to random number generator for all stochastic operations */
        /** @cond INTERNAL */
        struct additionalMutationParam {
            int networkSize = -1;
        };
        /** @endcond */

    public:
        /** @cond INTERNAL */
        const unsigned int ni; /**< Number of individuals in the population (constant after initialization) */
        unsigned int jn; /**< Initial number of judgment nodes per individual */
        unsigned int jnf; /**< Number of judgment node function types available */
        unsigned int pn; /**< Initial number of processing nodes per individual */
        unsigned int pnf; /**< Number of processing node function types available */
        bool fractalJudgment; /**< Flag indicating whether judgment nodes use fractal-based edge patterns */
        std::vector<Network> individuals; /**< Vector containing all Network individuals in the population */
        float bestFit; /**< Fitness value of the best individual in the current population */
        std::vector<int> indicesElite; /**< Indices of elite individuals (protected from mutation) */
        float meanFitness = 0; /**< Mean fitness across all individuals in the population */
        float minFitness; /**< Minimum fitness value in the current population */
        int maxNetworkSize; 
        std::vector<int> nFeatureValues; /** stores the number of feature values */
        /** @endcond */

        /** @name Constructor */
        /** @{ */
        /**
         * @brief Constructs a Population with specified parameters and initializes all individuals.
         * 
         * @details
         * This constructor creates a complete GNP population by:
         * 1. Initializing the random number generator with the given seed
         * 2. Creating ni Network individuals, each with:
         *    - jn judgment nodes
         *    - pn processing nodes
         *    - Random initial topology and function assignments
         *    - Optional fractal edge patterns (if fractalJudgment is true)
         * 
         * All individuals share the same random generator (via shared_ptr) to ensure
         * reproducibility and coordinated randomness across the population.
         * 
         * @param seed Random seed for the generator
         * @param _ni Number of individuals to create in the population
         * @param _jn Initial number of judgment nodes per individual
         * @param _jnf Number of judgment node function types (determines feature selection)
         * @param _pn Initial number of processing nodes per individual
         * @param _pnf Number of processing node function types (determines action/output)
         * @param _fractalJudgment If true, judgment nodes use fractal-based edge patterns; if false, standard edge patterns 
         * (see boundaryMutationFractal() for more informations on fractal boundaries)
         * @param _nFeatureValues set the number of features values to distinguish between numerical and categorical data
                - for numerical features: set 0 at the i-th feature 
                - for categorical features: set the numbers of categories at feature position i. This will be the amount of outgoing edges of a judgment node 
                - default is an empty vector and all features are treated as numerical
         */
        Population(
                int seed,
                const unsigned int _ni,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf,
                bool _fractalJudgment,
                std::vector<int> _nFeatureValues = {}
                ):
            generator(std::make_shared<std::mt19937_64>(seed)),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment),
            nFeatureValues(_nFeatureValues)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(generator,jn,jnf,pn,pnf,fractalJudgment,nFeatureValues));
        }
    }
        /** @} */

        /** @name Member Functions */
        /** @{ */
        /**
         * @brief Initializes decision boundaries for all judgment nodes in all individuals.
         * 
         * @details
         * This method sets up the decision boundaries that divide the feature space for
         * every judgment node across the entire population. The initialization process:
         * 
         * 1. Iterates through all individuals in the population
         * 2. For each individual, examines all inner nodes (judgment and processing nodes)
         * 3. For judgment nodes (type "J"):
         *    - **Standard mode** (if fractalJudgment is false):
         *      - Sets uniformly spaced boundaries within [minF[f], maxF[f]]
         *    - **Fractal mode** (if fractalJudgment is true):
         *      - Generates random production rule parameters using randomParameterCuts()
         *      - Calculates fractal lengths based on k_d parameters
         *      - Sets boundaries according to the fractal pattern within [minF[f], maxF[f]]
         * 
         * @warning This method should be called once after population initialization and before fitness
         * evaluation to ensure all judgment nodes have valid decision boundaries matching
         * the feature value ranges of the problem domain.
         * 
         * @param minF Vector of minimum values for each feature dimension (indexed by node function f)
         * @param maxF Vector of maximum values for each feature dimension (indexed by node function f)
         * 
         * @note minF and maxF must have size ≥ jnf (cover all judgment node functions)
         * 
         */
        void setAllNodeBoundaries(std::vector<float>& minF, std::vector<float>& maxF){
            for(auto& network : individuals){
               for(auto& node : network.innerNodes){
                   if(node.type == "J"){
                       if(fractalJudgment == true){
                           node.productionRuleParameter = randomParameterCuts(node.k_d.first-1, generator);
                           std::vector<float> fractals = fractalLengths(node.k_d.second, sortAndDistance(node.productionRuleParameter));
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f], fractals);
                       }else {
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f]);
                       }
                   }
               } 
            }
        }

        /**
         * @brief Executes network traversal for all individuals on a complete dataset.
         * 
         * @details
         * This function calls the traversePath() method for each individual in the population,
         * allowing all networks to process the entire feature matrix X and generate decision
         * sequences. For each individual:
         * 
         * 1. Clears previous decisions (stored in member decisions) and resets node usage tracking
         * 2. Processes each row in X through the network
         * 3. Records the decision made for each input sample
         * 4. Tracks which nodes were used during traversal
         * 
         * This is primarily used for batch prediction or analysis of network behavior across
         * the population and not fitness evaluation. The fitness needs to be calculated afterwards using
         * the network member decisions.
         * 
         * The dMax parameter prevents infinite loops by limiting consecutive
         * judgment nodes that can be traversed before forcing a decision.
         * 
         * @param X Feature matrix where each inner vector represents one sample with multiple features
         * @param dMax Maximum consecutive judgment nodes allowed per decision (prevents infinite graph cycles)
         * 
         * @see Network::traversePath()
         */
        void callTraversePath(
                const std::vector<std::vector<float>>& X,
                int dMax
                ){
            for (auto& network : individuals){
                network.traversePath(X,dMax);
            }
        }

        /**
         * @brief Applies a generic fitness function to all individuals in the population.
         * 
         * @details
         * This is a template function that accepts any callable object (function, lambda,
         * functor) and applies it to each individual network. 
         *
         * @tparam FuncFitness Callable type that accepts Network& and evaluates fitness
         * @param func Fitness function to apply (must accept Network& parameter)
         * 
         * @note This is an internal template used by specialized fitness methods
         */
        template <typename FuncFitness>
        void applyFitness(FuncFitness&& func){
            for (auto& network : individuals){
                func(network);
            }
        }

        /** @cond INTERNAL */

        /**
         * @brief Evaluates all individuals using classification accuracy on a labeled dataset.
         * 
         * @details
         * This function evaluates the entire population on a supervised learning task by
         * applying the fitAccuracy() method to each individual. For each network:
         * 
         * 1. Executes the network on all samples in X
         * 2. Compares predictions with ground truth labels in y
         * 3. Calculates accuracy as: (correct predictions) / (total samples)
         * 4. Stores result in the individual's fitness member
         * 
         * Networks that exceed the judgment depth limit (dMax) are marked as invalid and
         * receive fitness of 0. The penalty parameter is passed but currently unused in
         * the implementation.
         * 
         * This method is suitable for classification problems where the fitness metric is
         * prediction accuracy on a labeled dataset.
         * 
         * @param X Feature matrix (rows are samples, columns are features)
         * @param y Target labels vector corresponding to each sample in X
         * @param dMax Maximum consecutive judgment nodes per decision (prevents infinite loops)
         * @param penalty Divisor for fitness reduction on constraint violations (currently unused)
         * 
         * @post All individuals have fitness values in range [0.0, 1.0] representing accuracy
         * @post Invalid individuals (exceeding dMax) have fitness = 0
         * 
         * @note Fitness of 1.0 indicates perfect classification
         * @see Network::fitAccuracy()
         */
        void accuracy(
                const std::vector<std::vector<float>>& X,
                const std::vector<int>& y,
                int dMax,
                int penalty
                ){
            applyFitness([=](Network& network){
                    network.fitAccuracy(X,y,dMax,penalty);
            });
        }
        /** @endcond */

        /**
         * @brief Evaluates all individuals in an OpenAI Gymnasium-compatible reinforcement learning environment.
         * 
         * @details
         * This method applies fitGymnasium() to the entire population as reinforcement learning agents in
         * a Gymnasium environment. 
         * 
         * @param env GymEnvWrapper object providing interface to the Gymnasium environment
         * @param dMax Maximum consecutive judgment nodes per decision (prevents infinite loops in graph traversal)
         * @param maxSteps Maximum episode length (prevents indefinite episodes)
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed 
         * @param worstFitness Fitness value assigned when networks violate constraints
         * @param seed Random seed for environment initialization (currently unused in implementation)
         * 
         * @see Network::fitGymnasium()
         */
        void gymnasium(
            GymEnvWrapper& env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            int seed
                ){

            for(auto& network : individuals){
                network.fitGymnasium(
                        env,
                        dMax,
                        maxSteps,
                        maxConsecutiveP,
                        worstFitness,
                        seed
                        );
            }
        }

        /**
         * @brief Evaluates all individuals on the CartPole balancing control problem.
         * 
         * @details
         * This method applies fitCartpole() to the entire population 
         *
         * @see Network::fitCartpole()
         * 
         * @param dMax Maximum consecutive judgment nodes per decision (prevents infinite loops)
         * @param penalty Divisor applied to fitness when constraints are violated
         * @param maxSteps Maximum episode length 
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed 
         * 
         */
        void cartpole(
                int dMax,
                int penalty,
                int maxSteps,
                int maxConsecutiveP
                ){
            applyFitness([=](Network& network){
                    network.fitCartpole(dMax,penalty,maxSteps,maxConsecutiveP);
            });
        }

        /**
         * @brief Performs tournament selection with elitism to create the next generation.
         * 
         * @details
         * This method implements tournament selection, a standard evolutionary algorithm
         * selection operator that balances selective pressure with diversity. The process:
         * 
         * **For each individual in the population (except elite)**:
         * 1. Randomly sample N individuals to form a tournament
         * 2. Select the individual with highest fitness from the tournament
         * 3. Add the winner to the new population
         * 
         * **Elitism** (preserving E best individuals):
         * 1. Identifies the E individuals with highest fitness
         * 2. Copies them unchanged to the new population
         * 3. Stores their indices in indicesElite for mutation protection
         * 
         * **Statistic calculations**:
         * - bestFit: Maximum fitness in population (including elite)
         * - meanFitness: Average fitness in population 
         * - minFitness: Minimum fitness in population
         * 
         * Tournament selection provides tunable selective pressure: larger N increases pressure
         * (stronger individuals more likely to be selected), while smaller N maintains diversity.
         * Elitism ensures best solutions are never lost, providing monotonic improvement guarantee.
         * 
         * @param N Tournament size (number of individuals per tournament)
         * @param E Elite size (number of best individuals to preserve unchanged)
         * 
         * @note N ≥ 2 for meaningful selection pressure
         * @note Population size remains constant at ni
         * 
         */
        void tournamentSelection(int N, int E){
            std::vector<Network> selection;
            selection.reserve(individuals.size()); 
            std::unordered_set<int> tournament;
            tournament.reserve(N);
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
            meanFitness = 0;
            minFitness = individuals[0].fitness;
            bestFit = individuals[0].fitness;
            maxNetworkSize = individuals[0].innerNodes.size();

            for(int i=0; i<individuals.size()-E; i++){
                if(individuals[i].innerNodes.size() > maxNetworkSize){
                    maxNetworkSize = individuals[i].innerNodes.size();
                }
                float bestFitTournament = std::numeric_limits<float>::lowest();
                int indexBestIndTournament = 0;
                tournament.clear();

                while(tournament.size()<N){ // set the tournament
                    int randomInt = distribution(*generator);
                    tournament.insert(randomInt);
                }
                for(int k : tournament){
                   if(individuals[k].fitness > bestFitTournament){
                       bestFitTournament = individuals[k].fitness;
                       indexBestIndTournament = k;
                   } 
                }
                selection.push_back(individuals[indexBestIndTournament]);
                meanFitness += bestFitTournament;
                if (bestFitTournament < minFitness) {
                    minFitness = bestFitTournament;
                }
                if (bestFitTournament > bestFit) {
                    bestFit = bestFitTournament;
                }
            }
            setElite(E, individuals, selection);
            individuals = std::move(selection);
            meanFitness /= individuals.size();
        }
        /**
         * @brief Identifies and preserves the elite individuals in the selection.
         * 
         * @details
         * This method extracts the E best individuals from the current population and
         * adds them to the selection vector, implementing elitism in the evolutionary process.
         * 
         * **Track elite indices**: Stores indices where elite individuals are placed
         *    in the selection vector (used later to protect them from mutation)
         * 
         * @param E Number of elite individuals to preserve
         * @param individuals Copy of current population (will be modified during extraction)
         * @param selection Reference to new population being constructed (elite will be appended)
         * 
         * @note Elite indices are used to protect elite from mutation operations
         */
        void setElite(int E, const std::vector<Network>& individuals, std::vector<Network>& selection){
            indicesElite.clear();
            
            std::vector<unsigned int> candidateIndices(individuals.size());
            std::iota(candidateIndices.begin(), candidateIndices.end(), 0);

            for(int counter = 0; counter < E; ++counter){
                float eliteFit = std::numeric_limits<float>::lowest();
                unsigned int bestCandIdx = 0;
                for(unsigned int c = 0; c < candidateIndices.size(); ++c){
                    unsigned int idx = candidateIndices[c];
                    if(individuals[idx].fitness > eliteFit){
                        eliteFit = individuals[idx].fitness;
                        bestCandIdx = c;
                    }
                }
                unsigned int eliteIndex = candidateIndices[bestCandIdx];
                candidateIndices.erase(candidateIndices.begin() + bestCandIdx); // erase on small int-vector, not Network-vector

                indicesElite.push_back(selection.size()); // because of push_back of elite the index is the old size
                selection.push_back(individuals[eliteIndex]);
                if(eliteFit > bestFit){bestFit = eliteFit;} // set bestFit, otherwise elite will be forgotten in bestFit calculation
            }
        }

        /**
         * @brief Applies edge mutation to all non-elite individuals in the population.
         * 
         * @details
         * This method implements edge mutation, a topology-modifying operator that changes
         * the connections between nodes in the GNP networks. For each non-elite individual (network) 
         * and each node:
         * 
         * 1. **Inner nodes** (judgment and processing nodes): apply edgeMutation()
         * with probability probInnerNodes
         * 
         * 2. **Start node**: apply edgeMutation() with probability probStartNode
         * 
         * Edge mutation allows the evolutionary process to explore different network topologies
         * by redirecting execution flow. 
         *
         * **Elite protection**: Elite individuals (tracked via indicesElite) are excluded from
         * mutation to preserve the best solutions found so far.
         *
         * @see Node::edgeMutation()
         * 
         * @param probInnerNodes Probability (in [0.0, 1.0]) that each edge of inner nodes will be mutated
         * @param probStartNode Probability (in [0.0, 1.0]) that the start node's edge will be mutated
         * @param justUsedNodes If true, only applies edge mutation to nodes that were used during traversal (node.used == true). 
         * If false, applies to all nodes regardless of usage.
         * @param adaptToEdgeSize If true, mutation probability is adapted based on the number of edges (e.g., more edges → lower mutation probability) to prevent excessive disruption in highly connected nodes. 
         * @note adaptToEdgeSize not holds for starnode, because it has only one edge and should be mutated with the same probability as nodes with few edges to allow topology changes.
         *
         * @warning tournamentSelection() must have been called to set indicesElite
         * 
         */
        void callEdgeMutation(float probInnerNodes, float probStartNode, bool justUsedNodes = false, int k = 0){
            for(int i=0; i<individuals.size(); i++){

                int N;
                if(k > 0){
                    N = individuals[i].countEdges(justUsedNodes);
                }
                else {
                    N = 0;
                }

                if(std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()){// preventing elite
                    for(auto& node : individuals[i].innerNodes){
                        if(justUsedNodes == true && node.used == false){
                            continue;
                        } else {
                            node.edgeMutation(probInnerNodes, individuals[i].innerNodes.size(), k, N);
                        }
                    }
                    individuals[i].startNode.edgeMutation(probStartNode, individuals[i].innerNodes.size(), k, N);
                 }
             }
        }

         /**
         * @brief Applies a generic boundary mutation function to all judgment nodes in non-elite individuals.
         * 
         * @details
         * This is a template method that provides a flexible interface for applying various
         * boundary mutation strategies to the population. For each non-elite individual:
         * 
         * 1. Iterates through all judgment nodes (type "J")
         * 2. Applies the provided mutation function to each judgment node
         * 
         * The mutation function receives:
         * - Reference to the node (for modifying boundaries)
         * - additionalMutationParam struct (containing network size or other context)
         * 
         * This template design allows implementing different boundary mutation strategies
         * (uniform, normal, fractal, adaptive) without code duplication.
         * 
         * **Elite protection**: Elite individuals are excluded from mutation.
         * 
         * @tparam FuncMutation Callable type that accepts (Node&, const additionalMutationParam&)
         * @param func Mutation function to apply to each judgment node
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         * @note This is an internal template used by specialized boundary mutation methods
         */
        template <typename FuncMutation>
        void applyBoundaryMutation(FuncMutation&& func, bool justUsedNodes = false) {
            for (int i = 0; i < individuals.size(); ++i) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()) {
                    additionalMutationParam amp;
                    amp.networkSize = individuals[i].innerNodes.size();
                    for (auto& node : individuals[i].innerNodes) {
                        if (node.type == "J") {
                           if (justUsedNodes == true) {
                                if (node.used == true) {
                                    func(node, amp, justUsedNodes);
                                }
                            } else { 
                           func(node, amp, justUsedNodes); 
                            }
                        }
                    }
                }
            }
        }

        /**
         * @brief Applies uniform boundary mutation to all judgment nodes in the population.
         * 
         * @details
         * This method mutates judgment node boundaries using uniform distribution sampling.
         * Each boundary can be shifted anywhere within its valid range (between adjacent boundaries)
         * with equal probability.
         * 
         * @see Node::boundaryMutationUniform()
         * 
         * @param probability Probability (in [0.0, 1.0]) that each boundary will be mutated
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         */
        void callBoundaryMutationUniform(const float probability, bool justUsedNodes = false){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes){ 
                node.boundaryMutationUniform(probability);
            });
        }
       
        /**
         * @brief Applies normal (Gaussian) boundary mutation to all judgment nodes in the population.
         * 
         * @details
         * This method mutates judgment node boundaries using normal distribution sampling
         * centered at current boundary values. 
         *
         * For each judgment node in each non-elite individual, boundaries are mutated according
         * to Node::boundaryMutationNormal(), which samples from N(current_boundary, sigma²).
         *
         * @see Node::boundaryMutationNormal()
         * 
         * @param probability Probability (in [0.0, 1.0]) that each boundary will be mutated
         * @param sigma Standard deviation of the normal distribution
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         * @note Smaller sigma → more conservative, larger sigma → more exploratory
         */
        void callBoundaryMutationNormal(const float probability, const float sigma, bool justUsedNodes){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes){
                node.boundaryMutationNormal(probability, sigma);
            });
        }

        /**
         * @brief Applies normal boundary mutation with sigma adapted to network size.
         * 
         * @details
         * This method implements adaptive boundary mutation where the mutation strength
         * decreases as network size increases. The adaptive sigma is calculated as:
         * 
         * sigmaNew = sigma × (1 / log(networkSize))
         * 
         * **Concept**: Larger networks have more parameters to tune, so individual parameter
         * changes should be more conservative to avoid disrupting complex learned structures.
         * The logarithmic scaling provides smooth adaptation across network sizes.
         * 
         * This adaptive approach balances exploration and exploitation: smaller networks can
         * explore broadly, while larger networks receive more refined adjustments.
         * 
         * @param probability Probability (in [0.0, 1.0]) that each boundary will be mutated
         * @param sigma Base standard deviation (will be scaled down based on network size)
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         * @note Effective for problems where network size evolves during optimization
         * @see Node::boundaryMutationNormal()
         */
        void callBoundaryMutationNetworkSizeDependingSigma(const float probability, const float sigma, bool justUsedNodes){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam& amp, bool justUsedNodes){
                float sigmaNew = sigma * (1/log(amp.networkSize));
                node.boundaryMutationNormal(probability, sigmaNew);
            });
        }

        /**
         * @brief Applies normal boundary mutation with sigma adapted to number of node edges.
         * 
         * @details
         * This method implements adaptive boundary mutation where the mutation strength
         * decreases as the number of outgoing edges increases. The adaptive sigma is calculated as:
         * 
         * sigmaNew = sigma × (1 / log(edgeCount))
         * 
         * **Concept**: Judgment nodes with more outgoing edges partition the feature space
         * into more intervals, creating finer-grained decision boundaries. These require more
         * careful adjustment to avoid disrupting the detailed partitioning structure.
         * 
         * This node-level adaptation is more fine-grained than network-level adaptation,
         * allowing heterogeneous mutation strengths within the same network 
         * (each node hase his own distribution).
         * 
         * @param probability Probability (in [0.0, 1.0]) that each boundary will be mutated
         * @param sigma Standard deviation (will be scaled down based on edge count)
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         * @note Particularly useful when networks have heterogeneous judgment node structures
         * @see Node::boundaryMutationNormal()
         */
        void callBoundaryMutationEdgeSizeDependingSigma(const float probability, const float sigma, bool justUsedNodes){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes){
                float sigmaNew = sigma * (1/log(node.edges.size()));
                node.boundaryMutationNormal(probability, sigmaNew);
            });
        }
        
        /**
         * @brief Applies fractal boundary mutation to all judgment nodes with fractal structure.
         * 
         * @details
         * This specialized mutation operator is designed for judgment nodes that use fractal-based
         * edge patterns. Instead of directly mutating boundaries, it mutates the underlying
         * production rule parameters that generate the fractal structure, then recalculates
         * all boundaries accordingly.
         * 
         * For each judgment node in each non-elite individual:
         * 1. Mutates production rule parameters uniformly within valid ranges
         * 2. Recalculates fractal lengths based on k_d parameters
         * 3. Regenerates all boundaries to match the new fractal pattern
         * 
         * @param probability Probability (in [0.0, 1.0]) that each production parameter will be mutated
         * @param minF Vector of minimum values for all features (used for boundary recalculation)
         * @param maxF Vector of maximum values for all features (used for boundary recalculation)
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * 
         * @warning Only applicable if fractalJudgment is enabled (fractalJudgment = True)
         * @see Node::boundaryMutationFractal()
         */
        void callBoundaryMutationFractal(const float probability, std::vector<float> minF, std::vector<float> maxF, bool justUsedNodes){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes){
                node.boundaryMutationFractal(probability, minF, maxF);
            });
        }

        /**
         * @brief Performs crossover (recombination) between pairs of individuals in the population.
         * 
         * @details
         * This method implements crossover, a genetic operator that exchanges
         * parts of the gene structure between parent networks to create offspring. The process:
         * 
         * **Pairing**:
         * 1. Shuffles all individual indices randomly
         * 2. Pairs adjacent individuals in the shuffled order (0-1, 2-3, 4-5, etc.)
         * 3. Skips pairs where either parent is elite (elite protection)
         * 
         * **Node exchange**:
         * 1. Determines maximum exchangeable nodes: min(parent1.size, parent2.size). This is only 
         * needed if parents have different network sizes caused by applying callAddDelNodes().
         * 2. For each node (up to max exchangeable nodes) given type:
         *  "uniform":
         *    - With passed probability: swaps nodes at that position
         *    - After swap: repairs any invalid edges (edges pointing to non-existent nodes)
         *  "onepoint": draw a random number from the genotype and exchange all nodes until this point
         *  "randomWidth": exchanges subnetworks of potentially different widths between parents:
         *    1. Identifies successor nodes (active subnetwork) in both parents using findSuccessorNodes()
         *    2. Creates swap maps (old index -> new index) for remapping nodes between parents
         *    3. Validates that exchanging subnetworks won't reduce networks below 2 inner nodes (invalid network)
         *    4. Swaps nodes up to min(successor1.size, successor2.size) between the subnetworks
         *    5. Handles overhang nodes (extra nodes in larger subnetwork):
         *       - Adds overhang nodes from larger to smaller parent (addOverhangNodes)
         *       - Remaps all node IDs and edges in both parents to maintain consistency
         *       - Deletes overhang nodes from the larger parent (deleteOverhangNodes)
         *    6. This allows crossover between networks of different effective widths while preserving structure and modularity.
         * 
         * **Edge repair rules**:
         * - For "uniform" and "onepoint": Only check edges for the smaller parent (edges may become invalid after receiving nodes)
         * - For "randomWidth": Check edges for the larger parent (due to node deletion creating potential dangling edges)
         * - changeFalseEdges() redirects any dangling edges to valid random nodes
         * - Prevents graph structure corruption after recombination
         * 
         * @param propability Probability (in [0.0, 1.0]) that each node position will be exchanged (used for "uniform" type only). Default is 1.
         * @param type Type of the crossover:
         *  - "uniform": selects each node and exchanges them with given probability
         *  - "onepoint": draws a random cutpoint from the genotype and exchanges all nodes until this point
         *  - "randomWidth": exchanges subnetworks of different widths where all succesor nodes of a randomly selected node are exchanged
         * 
         * @note tournamentSelection() must have been called to set indicesElite
         * @note Only nodes up to min(size1, size2) can be exchanged for "uniform" and "onepoint" due to position-based matching
         * @note For "randomWidth", parent networks must have more than 2 inner nodes to safely use changeFalseEdges()
         */

        void crossover(float propability = 1, std::string type = ""){

            std::bernoulli_distribution distributionBernoulli(propability);
            int nNodesToExchange;
            std::vector<unsigned int> inds;
            for(int i=0; i<individuals.size(); i++){
                inds.push_back(i);
            }
            std::shuffle(inds.begin(), inds.end(), *generator);
            for(int i=0; i<inds.size()-1; i+=2){ // for each individual pair 
                
                std::vector<int> nodesToExchange;

                bool parent1IsElite = std::find(indicesElite.begin(), indicesElite.end(), inds[i]) != indicesElite.end();
                bool parent2IsElite = std::find(indicesElite.begin(), indicesElite.end(), inds[i+1]) != indicesElite.end();

                // Both elite → skip entirely (no crossover between two elites)
                if(parent1IsElite && parent2IsElite){
                    continue;
                }

                auto& parent1 = individuals[inds[i]];
                auto& parent2 = individuals[inds[i+1]];

                // check parent sizes 
                bool parent1IsLarger;
                bool parent2IsLarger;
                if(parent1.innerNodes.size() > parent2.innerNodes.size()){
                    parent1IsLarger = true;
                    parent2IsLarger = false;
                } else if (parent2.innerNodes.size() > parent1.innerNodes.size()){
                    parent1IsLarger = false;
                    parent2IsLarger = true;
                } else {
                    parent1IsLarger = false;
                    parent2IsLarger = false;
                }

                if(type == "uniform"){
                    nNodesToExchange = std::min(parent1.innerNodes.size(), parent2.innerNodes.size());
                    // set nodesToExchange
                    for(int k=0; k<nNodesToExchange; k++){
                        bool result = distributionBernoulli(*generator);
                        if(result == true){
                            nodesToExchange.push_back(k);
                        }
                    }

                } else if (type == "onepoint") {
                    int maxNodesToExchange = std::min(parent1.innerNodes.size(), parent2.innerNodes.size());
                    std::uniform_int_distribution<int> distributionUniform(0, maxNodesToExchange-1);
                    nNodesToExchange = distributionUniform(*generator);
                    // set nodesToExchange
                    for(int k=0; k<nNodesToExchange; k++){
                        nodesToExchange.push_back(k);
                    }

                } else {
                    nNodesToExchange = 0;
                }

                // exchange nodes
                if(type == "randomWidth"){

                    bool result = distributionBernoulli(*generator);
                    if(result == false){
                        continue; // random width crossover should not be applied, skip to next pair
                    }

                    std::vector<int> successor1 = findSuccessorNodes(parent1); // getting the node indices of subnetwork
                    std::vector<int> successor2 = findSuccessorNodes(parent2); // getting the node indices of subnetwork

                    // prevent networks <= 2 inner nodes otherwise the crossover would delete to many nodes
                    if((int)successor1.size() - (int)successor2.size() >= (int)parent1.innerNodes.size() - 2 ||
                       (int)successor2.size() - (int)successor1.size() >= (int)parent2.innerNodes.size() - 2 ||
                       (successor1.size() == 1 && successor2.size() == 1) ) {
                        continue;
                    }

                    if(parent1IsElite){
                        // parent1 is elite → only parent2 receives genes
                        parent2.nCrossovers += 1;

                        int minSubNodes = std::min(successor1.size(), successor2.size());

                        // Build remap: successor1[i] → successor2[i] (for edges in copied nodes)
                        std::unordered_map<int, int> remapForCopied;
                        for(int j=0; j<minSubNodes; j++){
                            remapForCopied[successor1[j]] = successor2[j];
                        }

                        // Copy nodes from parent1 to parent2
                        for(int j=0; j<minSubNodes; j++){
                            parent2.innerNodes[successor2[j]] = parent1.innerNodes[successor1[j]];
                            parent2.innerNodes[successor2[j]].id = successor2[j];
                        }

                        // Remap edges in the copied nodes
                        std::vector<int> copiedIndices(successor2.begin(), successor2.begin() + minSubNodes);
                        parent2.remapNodeIdsAndEdges(remapForCopied, copiedIndices, false);

                        // Fix edges pointing to invalid nodes
                        parent2.changeFalseEdges();

                    } else if(parent2IsElite){
                        // parent2 is elite → only parent1 receives genes
                        parent1.nCrossovers += 1;

                        int minSubNodes = std::min(successor1.size(), successor2.size());

                        // Build remap: successor2[i] → successor1[i] (for edges in copied nodes)
                        std::unordered_map<int, int> remapForCopied;
                        for(int j=0; j<minSubNodes; j++){
                            remapForCopied[successor2[j]] = successor1[j];
                        }

                        // Copy nodes from parent2 to parent1
                        for(int j=0; j<minSubNodes; j++){
                            parent1.innerNodes[successor1[j]] = parent2.innerNodes[successor2[j]];
                            parent1.innerNodes[successor1[j]].id = successor1[j];
                        }

                        // Remap edges in the copied nodes
                        std::vector<int> copiedIndices(successor1.begin(), successor1.begin() + minSubNodes);
                        parent1.remapNodeIdsAndEdges(remapForCopied, copiedIndices, false);

                        // Fix edges pointing to invalid nodes
                        parent1.changeFalseEdges();

                    } else { // no elite --> bidirectional swap

                        // swap map from individual1 to individual2
                        std::unordered_map<int, int> swapMap1 = initNodeSwapMap(successor1, successor2, parent2.innerNodes.size());
                        // swap map from individual2 to individual1
                        std::unordered_map<int, int> swapMap2 = initNodeSwapMap(successor2, successor1, parent1.innerNodes.size()); 

                        parent1.nCrossovers += 1;
                        parent2.nCrossovers += 1;

                        // exchange all nodes until same subnetwork size is reached 
                        int minSubNodes = std::min(successor1.size(), successor2.size());
                        for(int j=0; j<minSubNodes; j++){ 
                            std::swap(parent1.innerNodes[successor1[j]], parent2.innerNodes[successor2[j]]);
                        }

                        // add overhang nodes
                        if(successor1.size() > successor2.size()){
                           addOverhangNodes(successor1, successor2, parent1, parent2);
                        } else if (successor1.size() < successor2.size()) {
                           addOverhangNodes(successor2, successor1, parent2, parent1);
                        }

                        // initialize swap maps for remapping nodes and edges of both individuals
                        std::vector<int> indices1;
                        indices1.reserve(swapMap1.size()); 
                        for (auto const& [key, val] : swapMap1) {
                            indices1.push_back(val);
                        }

                        std::vector<int> indices2;
                        indices2.reserve(swapMap2.size()); 
                        for (auto const& [key, val] : swapMap2) {
                            indices2.push_back(val);
                        }

                        // remap nodes and edges of both individuals according to the swap maps 
                        parent1.remapNodeIdsAndEdges(swapMap2, indices2, false);
                        parent2.remapNodeIdsAndEdges(swapMap1, indices1, false);

                        // delete overhang nodes
                        if(successor1.size() > successor2.size()){
                            deleteOverhangNodes(successor1, successor2, parent1);
                        } else if (successor1.size() < successor2.size()) {
                            deleteOverhangNodes(successor2, successor1, parent2);
                        }

                        parent1.changeFalseEdges();
                        parent2.changeFalseEdges();
                    }
                   
                } else {

                    for(int k : nodesToExchange){
                        if(parent1IsElite){
                            // Elite donates: only parent2 receives genes from parent1
                            parent2.innerNodes[k] = parent1.innerNodes[k];
                        } else if(parent2IsElite){
                            // Elite donates: only parent1 receives genes from parent2
                            parent1.innerNodes[k] = parent2.innerNodes[k];
                        } else {
                            // Neither elite → normal bidirectional swap
                            std::swap(parent1.innerNodes[k], parent2.innerNodes[k]);
                        }
                    }

                    // repair node edges (only for the non-elite recipient or larger parent, as they may have received nodes that create invalid edges)
                    if(parent1IsElite){
                        parent2.changeFalseEdges();
                    } else if(parent2IsElite){
                        parent1.changeFalseEdges();
                    } else if(parent2IsLarger){
                        parent1.changeFalseEdges();
                    } else if (parent1IsLarger) {
                        parent2.changeFalseEdges(); 
                    }
                }
            }
        }

        /**
         * @brief Initializes a node index mapping between two subnode vectors for crossover operations.
         * 
         * @details Creates a mapping from subnodes1 indices to subnodes2 indices. If subnodes1 is longer,
         * the excess nodes are mapped to new indices starting from sizeNetwork2. This ensures proper node
         * index translation when combining networks during genetic crossover.
         * 
         * @param subnodes1 Vector of node indices from the first network
         * @param subnodes2 Vector of node indices from the second network
         * @param sizeNetwork2 The current size of the second network, used as base for new indices
         * @return std::unordered_map<int, int> Mapping from subnodes1 indices to corresponding target indices
         */
        std::unordered_map<int , int>initNodeSwapMap(
                const std::vector<int>& subnodes1, 
                const std::vector<int>& subnodes2,
                int sizeNetwork2
                ){
            // initialize swap map
            std::unordered_map<int, int> map;
            int minSubNodes = std::min(subnodes1.size(), subnodes2.size());
            for(int i=0; i<minSubNodes; i++){
                map[subnodes1[i]] = subnodes2[i];
            }

            if(subnodes1.size() <= minSubNodes){
                return map;
            } else { // adding overhang as appended indices
                for(int i = subnodes2.size(); i<subnodes1.size(); i++){
                    map[subnodes1[i]] = sizeNetwork2;
                    sizeNetwork2 ++;
                }
                return map;
            }
        }

         /**
         * @brief Adds overhang nodes from the larger parent subnetwork to the smaller parent subnetwork during crossover.
         * 
         * @details When two parents have different numbers of successor nodes, this method transfers
         * the excess nodes (overhang) from the larger parent's network to the smaller parent's network.
         * The overhang nodes are moved from parent1 to parent2's innerNodes vector.
         * 
         * @param successor1 Vector of successor node indices from the larger parent subnetwork
         * @param successor2 Vector of successor node indices from the smaller parent subnetwork
         * @param parent1 The parent network from which overhang nodes are extracted
         * @param parent2 The parent network to which overhang nodes are added
         *
         * @warning successor1 and successor2 must be sorted in ascending order
         * @note The added node IDs and edges may need further adjustment to maintain graph validity.
         */
        void addOverhangNodes(
                const std::vector<int>& successor1,// larger subnetwork 
                const std::vector<int>& successor2,
                Network& parent1, 
                Network& parent2
                ){

            int overhang = successor1.size() - successor2.size();
            for(int i=0; i<overhang; i++){
                int nodeIndex = successor1[successor2.size()+i];
                parent2.innerNodes.push_back(std::move(parent1.innerNodes[nodeIndex]));
                //std::cout << "added node " << nodeIndex << " from parent1 to parent2" << std::endl;
            }
        }

         /**
        * @brief Deletes overhang nodes from the larger parent subnetwork.
        * 
        * @details Removes excess nodes from parent1 when it has more successor subnodes than parent2.
        * For each overhang node to be deleted, this method:
        * 1. Creates a deletion map that remaps indices of nodes after the deleted node (shifting them down by one)
        * 2. Assigns the deleted node index a random valid edge 
        * 3. Remaps all node IDs and edges in the network to maintain consistency after deletion
        * 4. Erases the node from the innerNodes vector
        * This process ensures that all references remain valid after node removal, preventing dangling references.
        * 
        * @note The parent1 network must have more than 2 inner nodes, otherwise changeEdge() will cause an error
        *       when trying to assign a random valid edge for the deleted node.
        * 
        * @param successor1 Vector of successor node indices from the larger parent subnetwork
        * @param successor2 Vector of successor node indices from the smaller parent subnetwork
        * @param parent1 The larger parent subnetwork from which overhang nodes will be deleted (modified in-place)
        *
         * @warning successor1 and successor2 must be sorted in ascending order
        */
        void deleteOverhangNodes(
                const std::vector<int>& successor1,// larger subnetwork 
                const std::vector<int>& successor2,
                Network& parent1 
                ){

            int overhang = successor1.size() - successor2.size();
            for(int i=0; i<overhang; i++){

                // delete overhang nodes from parent1
                int nodeIndex = successor1[successor1.size()-1-i]; // node index for deletion

                parent1.innerNodes.erase(parent1.innerNodes.begin() + nodeIndex);
                //std::cout << "deleted node " << nodeIndex << " from parent1" << std::endl;

                // initialize deletion map
                std::unordered_map<int, int> map;
                for(int i=nodeIndex; i<parent1.innerNodes.size()+1; i++){
                    map[i+1] = i;
                }
                // random number for deleted node
                map[nodeIndex] = parent1.innerNodes[0].changeEdge(parent1.innerNodes.size(), nodeIndex);
                // remap nodes
                std::vector<int> indices; 
                for(int k=0; k<parent1.innerNodes.size(); k++){// all node must be checked for remaping 
                    indices.push_back(k);
                }
                parent1.remapNodeIdsAndEdges(map,indices,true);
            }
            parent1.innerNodes.shrink_to_fit(); // release excess capacity to prevent memory bloat
        }

        /**
         * @brief Find all successor nodes reachable after path traversal from a given start node.
         *
         * @details Starting from @p subNodesStart, this method collects successor nodes by
         * comparing traverse counters. Nodes whose traverse counter is greater than that of the
         * start node are considered successors. If the start node is unused, only the start node
         * itself is returned. The resulting indices are sorted in ascending order.
         *
         * @param individual The individual (network) whose inner nodes are traversed.
         * @param subNodesStart Index of the starting node in @c individual.innerNodes.
         *                      If -1, a random index is selected uniformly from the available inner nodes.
         * @param nSelectedNodes Maximum number of successor nodes to collect (excluding the start node).
         *                       If -1, a random count between 1 and the number of inner nodes minus one is chosen.
         * @return A sorted vector of node indices comprising the start node and up to
         *         @p nSelectedNodes successors.
         */
        std::vector<int> findSuccessorNodes(auto& individual, int subNodesStart = -1, int nSelectedNodes = -1){

            if(subNodesStart == -1){
                std::uniform_int_distribution<int> distributionUniform(0, individual.innerNodes.size()-1);
                subNodesStart = distributionUniform(*generator);
            }

            if(nSelectedNodes == -1){
                std::uniform_int_distribution<int> distributionUniform(1, individual.innerNodes.size()-1);
                nSelectedNodes = distributionUniform(*generator);
            }

            std::vector<int> nodeIndices;
            if (individual.innerNodes[subNodesStart].used == false){
                nodeIndices.push_back(subNodesStart);
                return nodeIndices; // if the node is unused, no successor nodes can be found
            }

            nodeIndices.push_back(subNodesStart);
            int traverseCounterStart = individual.innerNodes[subNodesStart].traverseCounter;
            for(int i=0; i<individual.innerNodes.size(); i++){
                int traverseCounterNode = individual.innerNodes[i].traverseCounter;
                if(traverseCounterNode > traverseCounterStart){
                    nodeIndices.push_back(i);
                    nSelectedNodes --;
                    if (nSelectedNodes == 0){
                        break;
                    }
                }
            }
            std::sort(nodeIndices.begin(), nodeIndices.end());
            return nodeIndices;
        }

        /**
         * @brief Applies node addition and deletion to individuals in the population.
         * 
         * @details
         * This method allowing the network topology to grow and shrink during evolution. 
         * For each individual, the addDelNodes() method decides whether to add or delete a node. 
         * 
         * **Addition**:
         * - Adds either a judgment node or processing node (based on pnf/(pnf+jnf) ratio)
         * - Restriction: Only adds a new node if all current nodes are traversed during the 
         *   transition path (the nodes flag "used" = true).
         * 
         * **Deletion**:
         * - Removes one unused node
         * - Updates all node IDs and edge connections to maintain graph validity
         *   Restriction: only delete a node if the node is **not** traversed during the 
         *   transition path (the node flag "used" = false).
         * - Enables network pruning to reduce complexity
         * 
         * This operator allows GNP to automatically discover appropriate network sizes,
         * and evolving toward optimal complexity for the problem. This extansion of GNP 
         * is called **variable-size** Genetic Network Programming. 
         *
         * @note See also our proposed operator in: 
         * "Variable-Size Genetic Network Programming for Portfolio Optimization with Trading Rules"
         * by Fabian Köhnke & Christian Borgelt, EvoApplications 2025 
         * https://doi.org/10.1007/978-3-031-90062-4_18
         * 
         * @param minF Vector of minimum values for all features (for new judgment node initialization)
         * @param maxF Vector of maximum values for all features (for new judgment node initialization)
         * @junk ratio of protected unused nodes (junk DNA). A value of 0.1 protects 10% of unused nodes 
         * and at least one node is always protected. 
         * 
         * @note This operator has not influence on the individuals fitness  
         * @see Network::addDelNodes()
         */
        void callAddDelNodes(std::vector<float>& minF, std::vector<float>& maxF, float junk=0){
            for(auto& ind : individuals){
                ind.addDelNodes(minF, maxF, junk, nFeatureValues);

            }
        }

        /**
         * @brief Evaluates all individuals across multiple seeds and stores per-seed rewards.
         * 
         * @details
         * Runs fitGymnasium() for each individual on each seed. The per-seed rewards
         * are stored in network.fitnessValues. The aggregated fitness is stored in
         * network.fitness as the mean reward across all seeds.
         *
         * @param env GymEnvWrapper object providing interface to the Gymnasium environment
         * @param dMax Maximum consecutive judgment nodes per decision
         * @param maxSteps Maximum episode length
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed
         * @param worstFitness Fitness value assigned when networks violate constraints
         * @param seeds Vector of random seeds for environment initialization
         */
        void gymnasiumMultiSeed(
            GymEnvWrapper& env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            const std::vector<int>& seeds
                ){

            for(auto& network : individuals){
                network.fitnessValues.clear();
                network.lastStepRewards.clear();
                float totalReward = 0.0f;

                for(int s : seeds){
                    network.fitGymnasium(env, dMax, maxSteps, maxConsecutiveP, worstFitness, s);
                    network.fitnessValues.push_back(network.fitness);
                    network.lastStepRewards.push_back(network.lastFitness);
                    totalReward += network.fitness;
                }

                // Default aggregation: mean reward
                network.fitness = totalReward / static_cast<float>(seeds.size());
            }
        }

        /**
         * @brief Calculates Pareto objectives (landing rate, mean reward) from fitnessValues.
         * 
         * @details
         * For each individual, extracts two objectives from the per-seed rewards 
         * stored in fitnessValues:
         *   - objectives[0] = landing rate (fraction of seeds with reward > landingThreshold)
         *   - objectives[1] = mean reward across all seeds
         *
         * The scalar fitness is set as: landingRate * 1000 + meanReward
         * This two-stage fitness ensures landing rate has priority while mean reward
         * breaks ties.
         *
         * @param landingThreshold Reward threshold above which a run counts as a landing (default: 100)
         *
         * @pre gymnasiumMultiSeed() must have been called to populate fitnessValues
         */
        void calculateParetoObjectives(float landingThreshold = 100.0f){
            for(auto& network : individuals){
                if(network.fitnessValues.empty()) continue;

                float totalReward = 0.0f;
                int landings = 0;

                for(size_t i = 0; i < network.fitnessValues.size(); i++){
                    totalReward += network.fitnessValues[i];
                    if(network.lastStepRewards[i] >= landingThreshold){  // ← EXAKT: letzter Step ≥ 100
                        landings++;
                    }
                }
              
                float meanReward = totalReward / static_cast<float>(network.fitnessValues.size());
                float landingRate = static_cast<float>(landings) / static_cast<float>(network.fitnessValues.size());

                // Store objectives: [0] = landing rate, [1] = mean reward
                network.objectives = {landingRate, meanReward};

                // Two-stage scalar fitness for elitism sorting
                network.fitness = landingRate * 10000.0f + meanReward;
            }
        }

        /**
         * @brief Checks if objectives A dominate objectives B (Pareto dominance).
         * 
         * @details
         * A dominates B if A is >= B in ALL objectives AND strictly > in at least one.
         *
         * @param a First objective vector
         * @param b Second objective vector
         * @return true if a dominates b
         */
        static bool dominates(const std::vector<float>& a, const std::vector<float>& b){
            bool strictlyBetter = false;
            for(size_t i = 0; i < a.size(); i++){
                if(a[i] < b[i]) return false;
                if(a[i] > b[i]) strictlyBetter = true;
            }
            return strictlyBetter;
        }

        /**
         * @brief Performs Pareto-based tournament selection with dual elitism.
         *
         * @details
         * Extends paretoTournamentSelection with two elite groups:
         * - E_reward best individuals by mean reward (exploitation)
         * - E_landing best individuals by landing rate (exploration)
         * This ensures that landing behavior is never lost through mutation.
         *
         * @param N Tournament size
         * @param E_reward Number of elite individuals by mean reward
         * @param E_landing Number of elite individuals by landing rate
         */
        void paretoTournamentSelection(int N, int E_reward, int E_landing){
            std::vector<Network> selection;
            selection.reserve(individuals.size());
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
            meanFitness = 0;
            minFitness = individuals[0].fitness;
            bestFit = individuals[0].fitness;
            maxNetworkSize = 0;

            int E = E_reward + E_landing;

            for(size_t i = 0; i < individuals.size() - E; i++){
                if (individuals[i].innerNodes.size() > maxNetworkSize){
                    maxNetworkSize = individuals[i].innerNodes.size();
                }
                // Build tournament
                std::unordered_set<int> tournament;
                tournament.reserve(N);
                while(static_cast<int>(tournament.size()) < N){
                    tournament.insert(distribution(*generator));
                }

                // Find non-dominated individuals in tournament
                std::vector<int> nonDominated;
                for(int idx : tournament){
                    bool isDominated = false;
                    for(int other : tournament){
                        if(other != idx &&
                           !individuals[idx].objectives.empty() &&
                           !individuals[other].objectives.empty() &&
                           dominates(individuals[other].objectives, individuals[idx].objectives)){
                            isDominated = true;
                            break;
                        }
                    }
                    if(!isDominated){
                        nonDominated.push_back(idx);
                    }
                }

                // Select random non-dominated individual
                std::uniform_int_distribution<int> ndDist(0, nonDominated.size()-1);
                int winner = nonDominated[ndDist(*generator)];

                selection.push_back(individuals[winner]);
                meanFitness += individuals[winner].fitness;
                if(individuals[winner].fitness < minFitness){
                    minFitness = individuals[winner].fitness;
                }
                if(individuals[winner].fitness > bestFit){
                    bestFit = individuals[winner].fitness;
                }
            }

            // Dual elitism
            setEliteDual(E_reward, E_landing, individuals, selection);
            individuals = std::move(selection);
            meanFitness /= individuals.size();
        }

        /**
         * @brief Identifies elite individuals by two criteria: mean reward AND landing rate.
         *
         * @param E_reward Number of elite by best fitness (mean reward)
         * @param E_landing Number of elite by best landing rate
         * @param individuals Copy of current population
         * @param selection Reference to new population being constructed
         */
        void setEliteDual(int E_reward, int E_landing,
                          const std::vector<Network>& individuals,
                          std::vector<Network>& selection){
            indicesElite.clear();

            // Track already selected indices to avoid duplicates
            std::unordered_set<int> alreadySelected;

            // Elite by best mean reward (fitness)
            for(int e = 0; e < E_reward; e++){
                float bestVal = std::numeric_limits<float>::lowest();
                int bestIdx = 0;
                for(int i = 0; i < individuals.size(); i++){
                    if(alreadySelected.count(i) == 0 && individuals[i].fitness > bestVal){
                        bestVal = individuals[i].fitness;
                        bestIdx = i;
                    }
                }
                indicesElite.push_back(selection.size());
                selection.push_back(individuals[bestIdx]);
                alreadySelected.insert(bestIdx);
                if(bestVal > bestFit) bestFit = bestVal;
            }

            // Elite by best landing rate (objectives[0])
            for(int e = 0; e < E_landing; e++){
                float bestVal = std::numeric_limits<float>::lowest();
                int bestIdx = 0;
                for(int i = 0; i < individuals.size(); i++){
                    if(alreadySelected.count(i) == 0 &&
                       !individuals[i].objectives.empty() &&
                       individuals[i].objectives[0] > bestVal){
                        bestVal = individuals[i].objectives[0];
                        bestIdx = i;
                    }
                }
                // Only add if actually has landings
                if(bestVal > 0.0f){
                    indicesElite.push_back(selection.size());
                    selection.push_back(individuals[bestIdx]);
                    alreadySelected.insert(bestIdx);
                } else {
                    // No lander found, fill with next best by fitness
                    float bestFitVal = std::numeric_limits<float>::lowest();
                    int bestFitIdx = 0;
                    for(int i = 0; i < individuals.size(); i++){
                        if(alreadySelected.count(i) == 0 && individuals[i].fitness > bestFitVal){
                            bestFitVal = individuals[i].fitness;
                            bestFitIdx = i;
                        }
                    }
                    indicesElite.push_back(selection.size());
                    selection.push_back(individuals[bestFitIdx]);
                    alreadySelected.insert(bestFitIdx);
                }
            }
        }

};

#endif
