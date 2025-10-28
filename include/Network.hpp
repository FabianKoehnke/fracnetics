#ifndef NETWORK_HPP
#define NETWORK_HPP
/// \cond INTERNAL
#include <random>
#include <utility>
#include <vector>
#include "Cartpole.hpp"
#include "Node.hpp"
#include "Fractal.hpp"
#include "GymnasiumWrapper.hpp"
/// \endcond

/**
 * @class Network
 * @brief Graph-based control structure for Genetic Network Programming (GNP).
 *
 * @details
 * The Network class represents the directed graph architecture used in 
 * **Genetic ``Network`` Programming (GNP)**, an evolutionary computation method 
 * based on graph structures rather than trees. A GNP network consists of:
 *
 * - **Judgment nodes** – evaluate conditional expressions based on input features
 * - **Processing nodes** – perform actions or output decisions
 * - **Directed edges** – define execution flow through the network
 *
 * Unlike traditional Genetic Programming (GP), GNP allows node reuse through
 * recurrent graph connections, enabling compact models with memory effects and
 * efficient decision-making. This makes GNP suitable for real-time control,
 * adaptive agents, reinforcement learning, and optimization problems.
 *
 * The network evolves over generations through evolutionary operators such as
 * mutation and crossover, gradually improving its performance based on a fitness
 * function.
 *
 * @note In the context of this implementation, the Network acts as the central
 * decision structure that interacts with input data while evolving its topology.
 * Furthermore, the networks can grow and shrink during the evolution using the
 * method addDelNodes(). 
 *
 * @note See also: "Variable-Size Genetic Network Programming for Portfolio Optimization
 * with Trading Rules" by Fabian Köhnke & Christian Borgelt, EvoApplications 2025
 * https://doi.org/10.1007/978-3-031-90062-4_18
 *
 * @nosubgrouping
 */
class Network {
    private:
        std::shared_ptr<std::mt19937_64> generator; ///< Shared pointer to random number generator for stochastic operations
    
    public:
        /** @cond INTERNAL */
        unsigned int jn; /**< Number of inital judgment nodes in the network */
        unsigned int jnf; /**< Number of judgment node function types available */
        unsigned int pn; /**< Number of inital processing nodes in the network */
        unsigned int pnf; /**< Number of processing node function types available */
        bool fractalJudgment; /**< Flag indicating whether outgoing edges follow a fractal pattern */
        std::vector<Node> innerNodes; /**< Collection of all judgment and processing nodes in the network */
        Node startNode; /**< Initial entry point for network execution */
        float fitness = std::numeric_limits<float>::lowest(); /**< Fitness value of the network (initialized to lowest possible value) */
        bool invalid = false; /**< Flag to indicate invalid individuals (e.g., exceeding judgment limits) */
        int currentNodeID; /**< ID of the currently active node during network traversal */
        int nConsecutiveP; /**< Counter for consecutive processing nodes encountered */
        int nUsedNodes; /**< Number of nodes that have been used during network traversal */
        std::vector<int> decisions; /**< Sequence of decisions made during network execution */
        /** @endcond */

        /** @name Constructor */
        /** @{ */
        /**
         * @brief Constructs a Network with specified parameters and initializes all nodes.
         * 
         * @details
         * This constructor creates a complete GNP network by initializing:
         * - A start node that serves as the entry point for network execution
         * - A specified number of judgment nodes with random function assignments
         * - A specified number of processing nodes with random function assignments
         * - Edges between nodes according to the specified pattern (standard or fractal)
         * 
         * For judgment nodes with fractal patterns enabled, the constructor calculates
         * the number of outgoing edges according to random_k_d_combination().
         * 
         * @param _generator Shared pointer to a Mersenne Twister random number generator (std::mt19937_64) used for all stochastic operations
         * @param _jn Number of initial judgment nodes in the network
         * @param _jnf Number of judgment node function types available (determines random function assignment range)
         * @param _pn Number of initial processing nodes in the network
         * @param _pnf Number of processing node function types available (determines random function assignment range)
         * @param _fractalJudgment If true, judgment nodes use fractal-based edge patterns; if false, standard edge patterns are used
         */
        Network(
                std::shared_ptr<std::mt19937_64> _generator,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf,
                bool _fractalJudgment
                ):
            generator(_generator),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment),
            startNode(generator,0,"S",0) // init start node
            
        {
            startNode.setEdges("S", jn+pn);
            std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
            for(int i=0; i<jn; i++){ // init judgment nodes 
                int randomInt = distributionJNF(*generator);
                innerNodes.push_back(Node(
                            generator, 
                            i, // node id 
                            "J", // node type 
                            randomInt // node function
                            ));
                if(fractalJudgment == false){
                    innerNodes.back().setEdges("J", pn+jn);
                }else{
                    std::pair<int, int> k_d = random_k_d_combination(pn+jn-1, generator);
                    innerNodes.back().k_d.first = k_d.first;
                    innerNodes.back().k_d.second = k_d.second;
                    innerNodes.back().setEdges("J", pn+jn, pow(k_d.first,k_d.second));
                }
            }
            std::uniform_int_distribution<int> distributionPNF(0, pnf-1);
            for(int i=jn; i<jn+pn; i++){ // init procesing nodes 
                int randomInt = distributionPNF(*generator);
                innerNodes.push_back(Node(
                            generator, 
                            i, // node id 
                            "P", // node type 
                            randomInt // node function
                            ));
                innerNodes.back().setEdges("P", jn+pn);
            }
        }
        /** @} */

        /** @name Member Functions */
        /** @{ */
        /**
         * @brief Resets the usage status of all nodes in the network.
         * 
         * @details
         * This method iterates through all inner nodes and sets their 
         * 'used' flag to false. This is typically called before a new traversal 
         * of the network to ensure accurate tracking of which nodes are visited 
         * during execution. The usage information is important for:
         * - Identifying unused nodes to delete them in addDelNodes() 
         * - Network analysis and optimization
         */
        void clearUsedNodes(){
            for(auto& node : innerNodes){
                node.used = false;
            }
        }
        
        /**
         * @brief Counts the number of nodes that have been marked as used and stores the result.
         * 
         * @details
         * This method iterates through all inner nodes (judgment and processing nodes) and counts how many 
         * have their 'used' flag set to true. The result is stored in the member 
         * variable nUsedNodes. This information is crucial for:
         * - Determining network efficiency (ratio of used to total nodes)
         * - Making decisions about node addition/deletion during evolution (addDelNodes())
         * 
         * @note This function should be called after a network traversal to get 
         *       accurate usage statistics.
         */
        void countUsedNodes(){
            nUsedNodes = 0;
            for(const auto& node : innerNodes){
                if(node.used == true){
                    nUsedNodes += 1;
                }
            }
        }
        
        /**
         * @brief Traverses the network for a complete dataset and records all decisions.
         * 
         * @details
         * This inline function executes the network for each row in the feature matrix X.
         * For each input vector, it:
         * 1. Clears the decisions vector and resets node usage flags (see clearUsedNodes())
         * 2. Initializes traversal at the start node's first edge
         * 3. Processes each judgment node (corresponding to a feature) and processing node through the network traversal
         * 4. Records the decision made by each traversed processing node 
         * 
         * @param X Feature matrix where each inner vector represents one sample with multiple features
         * @param dMax Maximum number of consecutive judgment nodes allowed before a decision must be made (prevents infinite loops)
         * 
         * @note The decisions vector will contain one integer decision per row in X
         * @note If dMax is exceeded during any decision, the invalid flag is set to true and the individual should be penalized 
         */
        void traversePath(
                const std::vector<std::vector<float>>& X,
                int dMax
                ){
           decisions.clear();
           clearUsedNodes();
           currentNodeID = startNode.edges[0];
           innerNodes[currentNodeID].used = true;
           nConsecutiveP = 0;
           invalid = false;
           int dec;
            for(const auto& row : X){
                dec = decisionAndNextNode(row, dMax);
                decisions.push_back(dec);
            }
        }

        /**
         * @brief Makes a single decision based on input data and transitions to the next node.
         * 
         * @details
         * This inline template function is the core decision-making mechanism of the network.
         * It processes a single data sample and navigates through the network graph until 
         * reaching a processing node, which provides the decision. The execution flow:
         * 
         * 1. **If current node is a Processing Node (P)**:
         *    - Returns the node's function value as the decision
         *    - Sets the next node via the node's outgoing and stores them in member currentNodeID
         *    - Increments the consecutive processing node counter
         * 
         * 2. **If current node is a Judgment Node (J)**:
         *    - Resets the consecutive processing node counter
         *    - Enters a loop traversing judgment nodes:
         *      - Evaluates the judgment condition using the specified feature. 
         *        The feature is specified by the node member f (function)
         *      - Follows the appropriate edge based on the judgment result (see judge())
         *      - Increments the judgment depth counter
         *      - If dMax is exceeded, marks the network as invalid and returns error code
         *    - Once a processing node is reached, returns its function value as decision
         * 
         * The function uses template parameters to accept various container types (std::vector,
         * std::array, etc.) for the input data, providing flexibility in usage.
         * 
         * @tparam dataContainer Type of the data container (must support operator[] for indexing)
         * @param data Input feature vector for the current sample (indexed by node function f)
         * @param dMax Maximum of consecutive judgment nodes before forcing termination
         * @return Integer decision value of the reached processing node (member f of processing node), or 
         *         std::numeric_limits<int>::lowest() if the network becomes invalid
         * 
         * @note Only works correctly for one-dimensional data access patterns (single data sample)
         * @warning Exceeding dMax sets the invalid flag to true and returns an error value
         */
        template <typename dataContainer> /** template for passing std::vector, std::array ...*/
        int decisionAndNextNode(const dataContainer& data, int dMax){
            int dec;
            int dSum = 0;
            double v;
            if(innerNodes[currentNodeID].type == "P"){
                dec = innerNodes[currentNodeID].f;
                // update currentNodeID to next node
                currentNodeID = innerNodes[currentNodeID].edges[0]; 
                innerNodes[currentNodeID].used = true;
                nConsecutiveP ++;

            } else if (innerNodes[currentNodeID].type == "J"){
                nConsecutiveP = 0;
                while(innerNodes[currentNodeID].type == "J"){
                    // update currentNodeID to next node
                    v = data[innerNodes[currentNodeID].f];
                    int judgeResult = innerNodes[currentNodeID].judge(v);
                    currentNodeID = innerNodes[currentNodeID].edges[judgeResult];
                    innerNodes[currentNodeID].used = true;
                    dSum ++;
                    if (dSum >= dMax){
                        invalid = true;
                        return std::numeric_limits<int>::lowest(); 
                    }
                }
                dec = innerNodes[currentNodeID].f;
                // update currentNodeID to next node
                currentNodeID = innerNodes[currentNodeID].edges[0]; 
                innerNodes[currentNodeID].used = true;
                nConsecutiveP ++;
           }
            return dec;
        }

        /** @cond INTERNAL */

        /**
         * @brief Evaluates network fitness using classification accuracy on a labeled dataset.
         * 
         * @details
         * This method evaluates the network on a supervised learning task
         * by executing it on each sample in the dataset and comparing predictions to ground truth.
         * The fitness calculation process:
         * 
         * 1. Initializes network state (clears used nodes, resets position to start node)
         * 2. For each sample in the dataset:
         *    - Executes decisionAndNextNode() to get a prediction
         *    - If the network becomes invalid (dMax exceeded), sets fitness to 0 and terminates
         *    - Compares the prediction with the true label
         *    - Increments correct counter if prediction matches label
         * 3. Calculates final fitness as accuracy: correct predictions / total samples
         * 
         * The function implements early stopping if the network becomes invalid at any point,
         * assigning zero fitness to encourage evolution away from such configurations.
         * 
         * @param X Feature matrix (rows are samples, columns are features)
         * @param y Target labels vector corresponding to each sample in X
         * @param dMax Maximum consecutive judgment nodes allowed per decision (prevents infinite loops)
         * @param penalty Divisor applied to fitness if constraints are violated (currently unused in this function)
         * 
         * @post The fitness member variable contains the classification accuracy (0.0 to 1.0)
         * @post The invalid flag indicates if the network exceeded judgment limits
         * @post Node usage flags reflect which nodes were visited during evaluation
         * 
         * @note Fitness of 0 indicates invalid network configuration
         * @note Fitness of 1.0 indicates perfect classification
         */
        
        void fitAccuracy(
                const std::vector<std::vector<float>>& X,
                const std::vector<int>& y,
                int dMax,
                int penalty
                ){

            clearUsedNodes();
            currentNodeID = startNode.edges[0];
            innerNodes[currentNodeID].used = true;
            int dec;
            invalid = false;
            float correct = 0;

            for(int i=0; i<y.size(); i++){
                int  dSum = 0; // to prevent dead-looks 
                dec = decisionAndNextNode(X[i], dMax);
                if(invalid == true){
                    fitness = 0;
                    break;
                }
                if(dec == y[i]){
                    correct += 1;
                }
            }

            if(invalid != true){
                fitness = correct / y.size();
            }
        }
        /** @endcond */


        /**
         * @brief Evaluates network fitness using an OpenAI Gymnasium-compatible environment.
         * 
         * @details
         * This method executes the network as a reinforcement learning agent in a
         * Gymnasium environment. 
         *
         * See also : https://gymnasium.farama.org
         *
         * The evaluation process simulates a complete episode:
         * 
         * 1. **Initialization**:
         *    - Resets the environment and obtains initial observation
         *    - Clears node usage tracking and resets network state
         *    - Initializes fitness accumulator and counters
         * 
         * 2. **Episode Loop** (until termination):
         *    - Network makes decision based on current observation
         *    - Checks validity constraints (see parameters dMax and maxConsecutiveP)
         *    - If constraints violated, assigns worst fitness and terminates
         *    - Executes action in environment via step() function
         *    - Accumulates reward into fitness
         *    - Updates observation for next iteration
         *    - Checks termination conditions (done flag or max steps reached)
         * 
         * 3. **Termination Conditions**:
         *    - Environment signals episode completion (done flag)
         *    - Maximum step limit reached
         *    - Network becomes invalid (exceeds dMax)
         *    - Too many consecutive processing nodes (exceeds maxConsecutiveP)
         * 
         * @param env GymEnvWrapper object providing the Gymnasium environment interface
         * @param dMax Maximum consecutive judgment nodes per decision (prevents infinite loops)
         * @param maxSteps Maximum number of environment steps per episode
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed.
         *  Here we can control the number of possible actions after using the observation data again.  
         * @param worstFitness Fitness value assigned when network violates constraints
         * @param seed Random seed for environment initialization 
         * 
         * @warning The network must produce valid actions for the specific Gymnasium environment
         */
        void fitGymnasium(
            GymEnvWrapper env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            int seed
            ){

            auto reset_out = env.reset();// Initial observation for the episode
            auto obs = reset_out[0].cast<std::vector<double>>();   
            auto info = reset_out[1];
            clearUsedNodes();
            currentNodeID = startNode.edges[0];
            innerNodes[currentNodeID].used = true;
            int dec;
            fitness = 0;
            nConsecutiveP = 0;
            invalid = false;
            bool done = false;
            int steps = 0;

            while(done == false){
                dec = decisionAndNextNode(obs, dMax);

                if (invalid || nConsecutiveP > maxConsecutiveP){
                    fitness = worstFitness;
                    break;
                }

                auto result = env.step(dec);
                obs = result[0].cast<std::vector<double>>(); 
                fitness += result[1].cast<float>();
                steps ++;
                if(result[2].cast<bool>() || steps >= maxSteps) done = true; 

            }
        }
          
        /**
         * @brief Evaluates network fitness on the CartPole balancing problem.
         * 
         * @details
         * This mthod implements a specialized fitness evaluation for the classic
         * CartPole control problem (similar to OpenAI Gymnasium's CartPole-v1). The CartPole
         * task requires balancing a pole on a moving cart through discrete left/right actions.
         *
         * See also: https://gymnasium.farama.org/environments/classic_control/cart_pole/
         * 
         * The evaluation process:
         * 
         * 1. **Initialization**:
         *    - Creates a new CartPole environment instance with the network's random generator
         *    - Resets network state and obtains initial observation (cart position, velocity, pole angle, angular velocity)
         *    - Initializes fitness counter, which increments for each successful step
         * 
         * 2. **Episode Loop**:
         *    - Increments fitness for each successful timestep
         *    - Executes current decision in CartPole environment
         *    - Obtains new observation from environment
         *    - Network makes next decision based on new observation
         *    - Checks termination conditions
         * 
         * 3. **Termination Conditions**:
         *    - Pole falls beyond recovery angle (environment sets terminated flag)
         *    - Maximum steps reached (episode length limit)
         *    - Network constraint violations:
         *      - Invalid flag set (judgment dMax exceeded)
         *      - Too many consecutive processing nodes (maxConsecutiveP exceeded)
         *      - Constraint violations apply penalty: fitness is divided by penalty factor
         * 
         * 4. **Fitness Calculation**:
         *    - Base fitness: number of steps the pole remained balanced
         *    - Penalized fitness: base fitness / penalty (if constraints violated)
         *    - Optimal fitness: maxSteps (indicates perfect balancing for entire episode)
         * 
         * @param dMax Maximum consecutive judgment nodes per decision (prevents infinite loops in graph traversal)
         * @param penalty Divisor applied to fitness when network violates structural constraints
         * @param maxSteps Maximum episode length (prevents indefinite episodes and caps maximum fitness)
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed 
         *  Here we can control the number of possible actions after using the observation data again.  
         * 
         */
        void fitCartpole(
            int dMax,
            int penalty,
            int maxSteps,
            int maxConsecutiveP
            ){

            clearUsedNodes();
            currentNodeID = startNode.edges[0];
            innerNodes[currentNodeID].used = true;
            int dec = 0;
            CartPole cp(generator);
            fitness = 0;
            nConsecutiveP = 0;
            invalid = false;
            std::array<double, 4> obs = cp.reset(); // Initial observation for the episode
            bool done = false;

            while(done == false){
                fitness ++;
                CartPole::StepResult result = cp.step(dec);
                obs = result.observation; 
                dec = decisionAndNextNode(obs, dMax);
                if(result.terminated || fitness >= maxSteps) done = true; 

                if (invalid || nConsecutiveP > maxConsecutiveP){
                    done = true;
                    fitness /= penalty;
                }
            }
        }

        /**
         * @brief Corrects invalid edge connections that point to non-existent nodes.
         * 
         * @details
         * This method validates and repairs the network's edge structure by ensuring
         * all edges point to valid node indices. This is necessary after structural mutations
         * such as node deletion (from addDelNodes()), where edges may become invalid. The repair process:
         * 
         * 1. Iterates through all nodes in the network
         * 2. For each node, examines all outgoing edges
         * 3. If an edge index exceeds the valid range (≥ innerNodes.size()), it indicates
         *    the edge points to a deleted or non-existent node
         * 4. Calls the node's changeEdge() method to randomly select a new valid target node
         * 
         * This ensures the network graph remains well-defined with all edges pointing
         * to existing nodes, preventing runtime errors during network traversal.
         * 
         * @note This function should be called after any operation that removes nodes
         */
        void changeFalseEdges(){
            for(auto& node : innerNodes){
                for(auto& edge : node.edges){
                    if(edge > innerNodes.size()-1){ // edge has no successor node -> set new one
                        edge = node.changeEdge(innerNodes.size(), edge);
                    }
                }
            }
        }

        /**
         * @brief Performs network grow and shrink during the evolution.
         * 
         * @details
         * This method implements structural mutation by adding or removing nodes from the network.
         * See also our proposed operator in: 
         * "Variable-Size Genetic Network Programming for Portfolio Optimization with Trading Rules"
         * by Fabian Köhnke & Christian Borgelt, EvoApplications 2025 
         * https://doi.org/10.1007/978-3-031-90062-4_18
         *
         * The mutation process:
         * 
         * **Decision Phase**:
         * - 50% probability to add a node vs. delete a node
         * - Determines node type (processing vs. judgment) based on pnf/(pnf+jnf) ratio
         * 
         * **Addition Branch** (if resultAdd == true):
         * - Condition: all nodes are currently used (nUsedNodes ≥ innerNodes.size() × 1)
         * - Processing Node Addition:
         *   - Assigns random function from [0, pnf-1]
         *   - Creates outgoing edge to random node in network
         *   - Increments pn counter
         * - Judgment Node Addition:
         *   - Assigns random function (feature index) from [0, jnf-1]
         *   - Sets edge structure (standard or fractal based on fractalJudgment flag)
         *   - For fractal mode: generates k-d combination, production rule parameters, fractal lengths
         *   - Sets edge boundaries based on feature value ranges [minF, maxF]
         *   - Increments jn counter
         * - Only one node added per call (break statement)
         * 
         * **Deletion Branch** (if resultAdd == false):
         * - Conditions:
         *   - More than one unused node exists (innerNodes.size() - nUsedNodes > 1)
         *   - Current node is unused (innerNodes[n].used == false)
         * - Deletion Process:
         *   1. Updates all node IDs greater than deleted node's ID (decrement by 1)
         *   2. Updates all edges pointing to nodes after deleted node (decrement by 1)
         *   3. Redirects edges pointing directly to deleted node using changeEdge()
         *   4. Updates start node edge if necessary
         *   5. Decrements jn or pn counter based on deleted node type
         *   6. Removes node from innerNodes vector
         * - Only one node deleted per call (break statement)
         * 
         * @param minF Vector of minimum feature values for each feature dimension (used for judgment node boundary initialization)
         * @param maxF Vector of maximum feature values for each feature dimension (used for judgment node boundary initialization)
         * 
         * @warning This method must be called bevore edgeMutation()! Reason: if edges are change 
         * by edgeMutation(), the node flag "used" is not guaranteed to be correct.
         *
         * @post All edges remain valid (no dangling edges)
         * @post Node IDs are contiguous from 0 to innerNodes.size()-1
         * 
         */
        void addDelNodes(std::vector<float>& minF, std::vector<float>& maxF){ 
            std::bernoulli_distribution distributionBernoulliAdd(0.5);
            float pnRatio = static_cast<float>(pnf) / static_cast<float>(pnf+jnf);
            std::bernoulli_distribution distributionBernoulliProcessingNode(pnRatio);
            bool resultAdd = distributionBernoulliAdd(*generator);
            countUsedNodes();
            for(int n=0; n<innerNodes.size(); n++){
                if(resultAdd && nUsedNodes >= innerNodes.size() * 1){// adding node hint 0.5 for more nodes in network
                    bool resultProcessingNode = distributionBernoulliProcessingNode(*generator);
                    if(resultProcessingNode){ // add processing node
                        std::uniform_int_distribution<int> distributionPNF(0, pnf-1);
                            int randomInt = distributionPNF(*generator);
                            innerNodes.push_back(Node(
                                        generator, 
                                        innerNodes.size(), // node id 
                                        "P", // node type 
                                        randomInt // node function
                                        ));
                            innerNodes.back().setEdges("P", innerNodes.size());
                            pn += 1;
                    }else{ // add judgment node 
                        std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
                        int randomInt = distributionJNF(*generator);
                        innerNodes.push_back(Node(
                                    generator, 
                                    innerNodes.size(), // node id 
                                    "J", // node type 
                                    randomInt // node function
                                    ));
                       
                        if(fractalJudgment == true){
                            std::pair<int, int> k_d = random_k_d_combination(pn+jn, generator); // normaly pn+jn-1 but jn counter comes later
                            innerNodes.back().k_d.first = k_d.first;
                            innerNodes.back().k_d.second = k_d.second;
                            innerNodes.back().setEdges("J", pn+jn, pow(k_d.first,k_d.second));
                            innerNodes.back().productionRuleParameter = randomParameterCuts(innerNodes.back().k_d.first-1, generator);
                            std::vector<float> fractals = fractalLengths(innerNodes.back().k_d.second, sortAndDistance(innerNodes.back().productionRuleParameter));
                            innerNodes.back().setEdgesBoundaries(minF[randomInt], maxF[randomInt], fractals);
                        }else {
                            innerNodes.back().setEdges("J", innerNodes.size());
                            innerNodes.back().setEdgesBoundaries(minF[randomInt], maxF[randomInt]);
                        }

                        jn += 1;
                    }
                    break; // NOTE: just one node can be added with break statement!

                }else if(!resultAdd && 
                        innerNodes.size()-nUsedNodes > 1 &&
                        innerNodes[n].used == false) // node is not used
                {// deleting nodes

                    for(auto& node : innerNodes){
                        // adapting node IDs of innerNodes
                        if(node.id > innerNodes[n].id){
                            node.id -= 1; // set back node numbers for nodes greater deleted id 
                        }
                    }

                    for(auto& node : innerNodes){ // for each node

                       // adapting node edges
                        for(int& edge : node.edges){ // for each edge

                            if(edge > n){
                                edge -= 1; // change edges to reset node ids 
                            }else if(edge == n){ // change edge pointing to deleted node
                                edge = node.changeEdge(innerNodes.size()-1, edge);
                            }
                        }
                    }

                    
                    // adapting start node edge; hint: no changeEdge() needed because a node connected 
                    // by a startnode ist always used. 
                    if(startNode.edges[0] > n){
                        startNode.edges[0] -= 1;
                    }

                    if(innerNodes[n].type == "J"){
                        jn -= 1;
                    }else if (innerNodes[n].type == "P") {
                        pn -= 1;
                    }

                    innerNodes.erase(innerNodes.begin()+n);
                    break; // TODO del this! 
                }
            }
        }
        /** @} */

};
#endif
