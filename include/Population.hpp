#ifndef POPULATION_HPP
#define POPULATION_HPP
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
         * 
         */
        Population(
                int seed,
                const unsigned int _ni,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf,
                bool _fractalJudgment
                ):
            generator(std::make_shared<std::mt19937_64>(seed)),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment)

    {
        for(int i=0; i<ni; i++){
            individuals.push_back(Network(generator,jn,jnf,pn,pnf,fractalJudgment));
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
            applyFitness([=](Network& network){
                    network.fitGymnasium(env,dMax,maxSteps,maxConsecutiveP,worstFitness,seed);
            });
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
            std::unordered_set<int> tournament;
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
            meanFitness = 0;
            minFitness = individuals[0].fitness;
            bestFit = individuals[0].fitness;

            for(int i=0; i<individuals.size()-E; i++){
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
                meanFitness += individuals[indexBestIndTournament].fitness;
                if (individuals[indexBestIndTournament].fitness < minFitness) {
                    minFitness = individuals[indexBestIndTournament].fitness;
                }
                if (individuals[indexBestIndTournament].fitness > bestFit) {
                    bestFit = individuals[indexBestIndTournament].fitness;
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
        void setElite(int E, std::vector<Network> individuals, std::vector<Network>& selection){
            unsigned int counter = 0;
            indicesElite.clear();
            while(counter<E){
                float eliteFit = individuals[0].fitness;
                unsigned int eliteIndex = 0;
                for(int i=1; i<individuals.size(); i++){
                    if(individuals[i].fitness > eliteFit){
                        eliteFit = individuals[i].fitness;
                        eliteIndex = i;
                    }
                }
                indicesElite.push_back(selection.size()); // because auf push_back of elite the index is the old size
                selection.push_back(individuals[eliteIndex]);
                individuals.erase(individuals.begin()+eliteIndex);
                counter += 1;
                if(eliteFit > bestFit){bestFit = eliteFit;} // set bestFit, otherwise elite will forgotten in bestFit calculation
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
         * 
         * @warning tournamentSelection() must have been called to set indicesElite
         * 
         */
        void callEdgeMutation(float probInnerNodes, float probStartNode){
            for(int i=0; i<individuals.size(); i++){
                if(std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()){// preventing elite
                    for(auto& node : individuals[i].innerNodes){
                        node.edgeMutation(probInnerNodes, individuals[i].innerNodes.size());
                    }
                    individuals[i].startNode.edgeMutation(probStartNode, individuals[i].innerNodes.size());
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
         * 
         * @note This is an internal template used by specialized boundary mutation methods
         */
        template <typename FuncMutation>
        void applyBoundaryMutation(FuncMutation&& func) {
            for (int i = 0; i < individuals.size(); ++i) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()) {
                    additionalMutationParam amp;
                    amp.networkSize = individuals[i].innerNodes.size();
                    for (auto& node : individuals[i].innerNodes) {
                        if (node.type == "J") {
                           func(node, amp); 
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
         * 
         */
        void callBoundaryMutationUniform(const float probability){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){ 
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
         * 
         * @note Smaller sigma → more conservative, larger sigma → more exploratory
         */
        void callBoundaryMutationNormal(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
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
         * 
         * @note Effective for problems where network size evolves during optimization
         * @see Node::boundaryMutationNormal()
         */
        void callBoundaryMutationNetworkSizeDependingSigma(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam& amp){
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
         * 
         * @note Particularly useful when networks have heterogeneous judgment node structures
         * @see Node::boundaryMutationNormal()
         */
        void callBoundaryMutationEdgeSizeDependingSigma(const float probability, const float sigma){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
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
         * 
         * @warning Only applicable if fractalJudgment is enabled (fractalJudgment = True)
         * @see Node::boundaryMutationFractal()
         */
        void callBoundaryMutationFractal(const float probability, std::vector<float> minF, std::vector<float> maxF){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&){
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
         * 2. For each node (up to max exchangeable nodes):
         *    - With passed probability: swaps nodes at that position
         *    - After swap: repairs any invalid edges (edges pointing to non-existent nodes)
         * 
         * **Edge repair rules**:
         * - Only check edges for the smaller parent (edges may become invalid after receiving nodes)
         * - changeFalseEdges() redirects any dangling edges to valid random nodes
         * - Prevents graph structure corruption after recombination
         * 
         * @param propability Probability (in [0.0, 1.0]) that each node position will be exchanged
         * 
         * @note tournamentSelection() must have been called to set indicesElite
         * @note Only nodes up to min(size1, size2) can be exchanged due to position-based matching
         */
        void crossover(float propability){
            std::bernoulli_distribution distributionBernoulli(propability);
            std::vector<unsigned int> inds;
            for(int i=0; i<individuals.size(); i++){
                inds.push_back(i);
            }
            std::shuffle(inds.begin(), inds.end(), *generator);
            for(int i=0; i<inds.size()-1; i+=2){ // for each individual
                if(std::find(indicesElite.begin(), indicesElite.end(), inds[i]) != indicesElite.end() ||
                    std::find(indicesElite.begin(), indicesElite.end(), inds[i+1]) != indicesElite.end()
                    ){ // preventing crossover for elite
                    continue;
                }
                auto& parent1 = individuals[inds[i]];
                auto& parent2 = individuals[inds[i+1]];
                int maxNodeNumbers = std::min(parent1.innerNodes.size(), parent2.innerNodes.size());

                for(int k=0; k<maxNodeNumbers-1; k++){ // for each node
                    bool result = distributionBernoulli(*generator);
                    if(result){
                        std::swap(parent1.innerNodes[k], parent2.innerNodes[k]);
                        // just check for "false edges" if parent is the smaller one (expensive)
                        if(parent1.innerNodes.size() < parent2.innerNodes.size()){
                            parent1.changeFalseEdges();
                        } else if (parent2.innerNodes.size() < parent1.innerNodes.size()) {
                            parent2.changeFalseEdges(); 
                        }
                    }
 
                }

            }

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
         * 
         * @note This operator has not influence on the individuals fitness  
         * @see Network::addDelNodes()
         */
        void callAddDelNodes(std::vector<float>& minF, std::vector<float>& maxF){
            for(auto& ind : individuals){
                ind.addDelNodes(minF, maxF);

            }
        }
        /** @} */

};

#endif
