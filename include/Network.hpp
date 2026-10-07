#ifndef NETWORK_HPP
#define NETWORK_HPP
/// \cond INTERNAL
#include <cstdint>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>
#include "Cartpole.hpp"
#include "Node.hpp"
#include <unordered_map>
#include <algorithm>
#include <queue>
#include <cassert>
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
        /** Diagnostic: identity of the RNG this network shares with its population. */
        std::uintptr_t rngPointer() const {
            return reinterpret_cast<std::uintptr_t>(generator.get());
        }

        /** @cond INTERNAL */
        unsigned int jn; /**< Number of inital judgment nodes in the network */
        unsigned int jnf; /**< Number of judgment node function types available */
        unsigned int pn; /**< Number of inital processing nodes in the network */
        unsigned int pnf; /**< Number of processing node function types available */
        bool fractalJudgment; /**< Flag indicating whether outgoing edges follow a fractal pattern */
        bool useExperience; ///< If true, judgment nodes are created as "JE" (experience-weighted)
        std::vector<Node> innerNodes; /**< Collection of all judgment and processing nodes in the network */
        Node startNode; /**< Initial entry point for network execution */
        float fitness = std::numeric_limits<float>::lowest(); /**< Fitness value of the network (initialized to lowest possible value) */
        float lastFitness = std::numeric_limits<float>::lowest(); /**< last Fitness value from episode (used for analysis) */ 
        float lastFitnessII = std::numeric_limits<float>::lowest(); /**< last Fitness value from episode (used for analysis) */ 

        // --- EMA-Fitness über Generationen hinweg (verhindert "Vergessen" alter Seed-Auswertungen) ---
        float alpha = 0.15f; /**< EMA-Glättungsfaktor: Gewicht der aktuellen Generation gegenüber der bisherigen Historie.
                                  Wird spaeter individuell (z.B. aehnlichkeitsbasiert bei Mutation/Crossover) angepasst. */
        float emaFitness = std::numeric_limits<float>::lowest(); /**< Laufender EMA-Fitnesswert ueber alle je berechneten Generationen dieses Individuums. */
        bool emaInitialized = false; /**< Flag: true sobald emaFitness mindestens einmal gesetzt wurde (verhindert Vermischung mit dem lowest()-Initialwert). */

        // --- Lineage-Fitness: inkrementeller Online-Mittelwert ueber ALLE jemals
        //     gesehenen Seeds dieser Linie (nicht nur die aktuelle Generation),
        //     vererbbar bei Crossover/Mutation (siehe blendLineageWith()). Feature
        //     ist rein additiv/optional: wird automatisch mitgefuehrt, beeinflusst
        //     aber standardmaessig KEINE bestehende Selektion (fitness/fitnessValues
        //     bleiben unveraendert) -- kann bei Bedarf zusaetzlich zur Sortierung
        //     herangezogen werden, ohne dass etwas anderes angepasst werden muss. ---
        static constexpr size_t N_OBJECTIVES = 5; /**< Size of lastEpisodeObjectives / lexicaseObjectives. */
        static constexpr float LANDING_SUCCESS_THRESHOLD = 100.0f; /**< Schwelle für "erfolgreich gelandet" (Gymnasium vergibt +100 Terminalbonus bei sicherer Landung, -100 bei Crash). */
        float lineageMean = 0.0f; /**< Inkrementeller (Welford) laufender Mittelwert der Rohreward-Fitness ueber ALLE je von dieser Linie gesehenen Seeds. */
        int lineageN = 0; /**< Anzahl der Beobachtungen, die in lineageMean eingeflossen sind. */
        float lineageM2 = 0.0f; /**< Welford-M2-Akkumulator (Summe quadrierter Abweichungen vom laufenden
                                      Mittelwert) fuer lineageMean -- Grundlage fuer die Varianz-/Unsicherheits-
                                      abschaetzung (lineageVariance()/lineageLCB()), analog zum bereits
                                      existierenden EdgeExperience::m2Return-Pattern in Node.hpp. */
        float lineageSuccessRate = 0.0f; /**< Inkrementelle laufende Landungserfolgsrate (Anteil Seeds mit fitness > LANDING_SUCCESS_THRESHOLD) ueber ALLE je gesehenen Seeds dieser Linie. */
        int lineageSuccessN = 0; /**< Anzahl der Beobachtungen, die in lineageSuccessRate eingeflossen sind. */

        bool invalid = false; /**< Flag to indicate invalid individuals (e.g., exceeding judgment limits) */
        int currentNodeID; /**< ID of the currently active node during network traversal */
        int nConsecutiveP; /**< Counter for consecutive processing nodes encountered */
        int nUsedNodes; /**< Number of nodes that have been used during network traversal */
        int nBest = 0; /**< counter for n best times of an individual during evolution */
        std::vector<int> decisions; /**< Sequence of decisions made during network execution */
        std::vector<float> fitnessValues = {}; /** placeholder for storing multiple fitness values */
        // ─── Per-seed traversal record (Population::gymnasiumMultiSeed(), Population::crossover(type="seedSpecialist")) ───
        // Entry s holds the sorted list of innerNodes-indices actually traversed (used) while
        // evaluating seeds[s] specifically -- unlike Node::used/traverseCounter, which accumulate
        // ACROSS all seeds of a generation (by design, e.g. for callAddDelNodes()), this vector
        // isolates the active sub-graph of EACH individual seed. Populated unconditionally by
        // gymnasiumMultiSeed() via a traverseCounter before/after diff, so it never interferes with
        // the existing cumulative used/traverseCounter semantics.
        std::vector<std::vector<int>> visitedNodesPerSeed = {};
        int traverseCounter = 0; /**< Counter for how many times the network has been traversed (used for analysis) */
        size_t nCrossovers = 0; /**< Counter for how many times the network has been involved in crossover (used for analysis) */
        // Per-Generation-Flag: wird von jeder Mutations-/Crossover-Operation, die diese
        // Linie TATSAECHLICH veraendert (nicht nur "probability check bestanden, aber
        // 0 Aenderungen ausgefuehrt"), auf true gesetzt. capLineageAfterMutation() liest
        // dieses Flag, um die lineageMean/lineageSuccessRate-Historie NUR bei Linien zu
        // deckeln, die diese Generation wirklich mutiert/gekreuzt wurden -- unveraenderte
        // Individuen sollen ihre akkumulierte Historie behalten. Wird nach dem Lesen in
        // capLineageAfterMutation() wieder auf false zurueckgesetzt.
        bool structureChangedThisGen = false;
        std::vector<float> objectives = {}; 
        std::vector<float> lastStepRewards = {};
        std::vector<float> lastStepRewardsII = {};

        // --- Epsilon-Lexicase-Selektion: 5-dimensionaler Ziel-Vektor pro Episode -----
        // (siehe Population::lexicaseSelection()). Zerlegt das Problem in isolierte,
        // voneinander unabhaengige Ziele, damit ein dominanter Treibstoff-Abzug nicht
        // laenger das fragile Balance-Signal ausloescht -- jedes Ziel wird von der
        // Selektion separat bewertet statt zu einem einzigen Skalar vermischt.
        // Reihenfolge: [0]=Haltungskontrolle (Balance), [1]=Sinkflug-Sicherheit
        // (Aufprallgeschwindigkeit), [2]=Horizontale Praezision (Pad-Treffer),
        // [3]=Treibstoff-Effizienz, [4]=Gym-Standard-Reward (holistisch). Fuer alle
        // 5 Ziele gilt: hoeher = besser (siehe fitGymnasium()).
        /** Objective vector of the last finished episode, filled by fitGymnasium().
         *
         *  Five criteria, EVERY ONE of which is maximised by landing on the pad:
         *    [0] total Gymnasium reward    -- holistic, and it already rewards landing
         *    [1] -|vy_end| if landed       -- descent safety (vertical)
         *    [2] -|vx_end| if landed       -- lateral speed, the one that makes a lander
         *                                     touch down softly and then slide off the pad
         *    [3] -|x_end|  if landed       -- horizontal precision
         *    [4] landed (1.0 / 0.0)        -- the goal itself
         *
         *  vx and vy are deliberately SEPARATE. Measured on a trained network over 1000
         *  seeds: successful landings end at a combined speed of exactly 0.00, failures at
         *  0.49, and the failures touch down ON the pad before sliding off it -- so what
         *  is left over is lateral. A combined -(|vx|+|vy|) lets a network score on the
         *  vertical part alone and ignore the horizontal one, which is what it did.
         *
         *  THE CONDITION ON [1]-[3] IS THE POINT. An earlier version measured them
         *  unconditionally in the final frame, and a network that hovered over the pad
         *  until the step limit won them all: it stands still (vy, vx near zero) above the
         *  pad (x near zero) and survives the longest. Four of five test cases went to a
         *  hoverer, and the population drifted there -- measured over 50 generations, the
         *  share of episodes reaching the step limit rose from 0.1% to 2.9% while the
         *  fitness fell from -375 to -592. Without a landing these three now yield a fixed
         *  penalty instead, so hovering can win none of them, and among the landings they
         *  resolve exactly the differences that decide pad or no pad.
         *
         *  The penalties are only slightly worse than the worst achievable real value, not
         *  hugely worse: epsilon-lexicase derives its tolerance from the spread of the
         *  values, and a far-away penalty would inflate that spread until the test case
         *  stops filtering at all.
         *
         *  Deliberately NOT included: the posture sum and the fuel sum used before. Both
         *  are sums of non-positive per-step terms, so both are maximised by an episode
         *  that ends at once without firing an engine -- they point away from the task and
         *  measurably stalled the evolution when used as lexicase test cases. */
        std::vector<float> lastEpisodeObjectives = std::vector<float>(5, 0.0f);
        std::vector<float> lexicaseObjectives = {}; /**< Objective vector averaged over all seeds, filled by gymnasiumMultiSeed()/gymnasium() -- basis for Population::lexicaseSelection(). */
        /** Per seed the 5-D objective vector of that episode, filled by gymnasiumMultiSeed().
         *  Unlike lexicaseObjectives it keeps the seeds apart, which turns the objectives
         *  and the N seeds into 5*N graded test cases for
         *  Population::lexicaseSelection(type="objectivesPerSeed"). */
        std::vector<std::vector<float>> objectivesPerSeed = {};

        bool frozenExperience = false;

        // ─── Experience Episode Logging (für JE-Nodes) ────────────────────────

        /**
         * @brief Records one JE-node activation within a single env step.
         */
        struct JEVisit {
            int   nodeID;    ///< Which JE node was active
            int   edgeIndex; ///< Which edge was chosen by judgeWithExperience()
            float obsF;      ///< obs[node.f] at the time of the decision
        };

        /**
         * @brief Stores all JE activations and the resulting step reward for one env step.
         */
        struct StepLog {
            std::vector<JEVisit> jeVisits; ///< All JE nodes visited during this step
            float reward;                  ///< Reward received after env.step()
        };

        std::vector<StepLog> episodeLog;         ///< Accumulated per-step log; cleared after backward pass
        std::vector<JEVisit> currentStepJEVisits; ///< Populated during decisionAndNextNode(); flushed per step


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
         * @param _nFeatureValues set the number of features values to distinguish between numerical and categorical data
         *          - for numerical features: set 0 at the i-th feature 
         *          - for categorical features: set the numbers of categories at feature position i. This will be the amount of outgoing edges of a judgment node 
         *          - default is an empty vector and all features are treated as numerical
         */
        Network(
                std::shared_ptr<std::mt19937_64> _generator,
                unsigned int _jn,
                unsigned int _jnf,
                unsigned int _pn,
                unsigned int _pnf,
                bool _fractalJudgment,
                bool _useExperience = false,
                std::vector<int> _nFeatureValues = {}
                ):
            generator(_generator),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment),
            useExperience(_useExperience),
            startNode(generator,0,"S",0)
        {
            startNode.setEdges("S", jn+pn);
            std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
            int nOutgoingEdges;
            std::string jNodeType = useExperience ? "JE" : "J";
            for(int i=0; i<jn; i++){
                int randomInt = distributionJNF(*generator);
                innerNodes.push_back(Node(
                            generator,
                            i,
                            jNodeType,
                            randomInt
                            ));
                // Alle Individuen starten mit derselben Topologie (jn Judgment-, dann pn
                // Processing-Knoten), also bekommt Knoten i in JEDEM Individuum dieselbe
                // innovationID -- genau wie in NEAT, wo die Startpopulation eine gemeinsame
                // Struktur und damit gemeinsame historische Marker teilt. Nur so sind die
                // Startnetze ueberhaupt aneinander ausrichtbar.

                if(_nFeatureValues.size()>0){
                     nOutgoingEdges = _nFeatureValues[randomInt];
                } else {nOutgoingEdges = 0;}

                if(fractalJudgment == false || nOutgoingEdges != 0){
                    innerNodes.back().setEdges("J", pn+jn, nOutgoingEdges);
                }else{
                    std::pair<int, int> k_d = random_k_d_combination(pn+jn-1, generator);
                    innerNodes.back().k_d.first = k_d.first;
                    innerNodes.back().k_d.second = k_d.second;
                    innerNodes.back().setEdges("J", pn+jn, pow(k_d.first,k_d.second));
                }
                if(useExperience) innerNodes.back().initEdgeExperience();
            }
            std::uniform_int_distribution<int> distributionPNF(0, pnf-1);
            for(int i=jn; i<jn+pn; i++){
                int randomInt = distributionPNF(*generator);
                innerNodes.push_back(Node(generator, i, "P", randomInt));
                innerNodes.back().setEdges("P", jn+pn);
            }
            // Start tracking transitions right away so that findTransitionClusters()
            // addDelNodes()/addOverhangNodes()/deleteOverhangNodes() keep this matrix
            // in sync as innerNodes grows/shrinks afterwards.
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
        /**
         * @brief Advances the per-node grace-period counters by one generation.
         *
         * @details
         * Must be called ONCE per generation while `used` still holds the flags of the
         * generation that just finished -- i.e. after the evaluation and before the next
         * one clears them (fitGymnasium(newRun=true) calls clearUsedNodes()).
         * Nodes that have never been traversed are left untouched, so they stay
         * immediately deletable; only nodes that were active at some point age.
         *
         * @see addDelNodes(), Node::unusedSince
         */
        void ageUnusedNodes(){
            for(auto& node : innerNodes){
                if(node.used){
                    node.everUsed = true;
                    node.unusedSince = 0;
                } else if(node.everUsed && node.unusedSince < std::numeric_limits<int>::max()){
                    node.unusedSince += 1;
                }
            }
        }

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
           innerNodes[currentNodeID].traverseCounter += 1;
           innerNodes[currentNodeID].lastVisitStep = traverseCounter;
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
        template <typename dataContainer>
        int decisionAndNextNode(const dataContainer& data, int dMax){
            int dec;
            int dSum = 0;
            double v;

            currentStepJEVisits.clear(); // reset JE visit log for this step

            if(innerNodes[currentNodeID].type == "P"){
                dec = innerNodes[currentNodeID].f;
                int prevNodeID = currentNodeID;
                currentNodeID = innerNodes[currentNodeID].edges[0];
                innerNodes[currentNodeID].used = true;
                traverseCounter++;
                innerNodes[currentNodeID].traverseCounter += 1;      // wie OFT der Knoten betreten wurde
                innerNodes[currentNodeID].lastVisitStep = traverseCounter; // WANN zuletzt (Reihenfolge)
                nConsecutiveP++;

            } else if (innerNodes[currentNodeID].type == "J"
                    || innerNodes[currentNodeID].type == "JE") {

                nConsecutiveP = 0;

                while(innerNodes[currentNodeID].type == "J"
                   || innerNodes[currentNodeID].type == "JE") {

                    v = data[innerNodes[currentNodeID].f];
                    int judgeResult;

                    if (innerNodes[currentNodeID].type == "JE") {
                        // Experience-weighted decision 
                        //judgeResult = innerNodes[currentNodeID].judgeWithExperience(
                        //                  static_cast<float>(v));
                        judgeResult = innerNodes[currentNodeID].judge(static_cast<float>(v));
                        // Log visit for backward return pass
                        currentStepJEVisits.push_back({
                            static_cast<int>(currentNodeID),
                            judgeResult,
                            static_cast<float>(v)
                        });
                    } else {
                        // Standard reactive decision
                        judgeResult = innerNodes[currentNodeID].judge(v);
                    }

                    int prevNodeID = currentNodeID;
                    currentNodeID = innerNodes[currentNodeID].edges[judgeResult];
                    innerNodes[currentNodeID].used = true;
                    traverseCounter++;
                    innerNodes[currentNodeID].traverseCounter += 1;
                    innerNodes[currentNodeID].lastVisitStep = traverseCounter;
                    dSum++;
                    if (dSum >= dMax){
                        invalid = true;
                        return std::numeric_limits<int>::lowest();
                    }
                }

                dec = innerNodes[currentNodeID].f;
                int prevNodeID = currentNodeID;
                currentNodeID = innerNodes[currentNodeID].edges[0];
                innerNodes[currentNodeID].used = true;
                traverseCounter++;
                innerNodes[currentNodeID].traverseCounter += 1;
                innerNodes[currentNodeID].lastVisitStep = traverseCounter;
                nConsecutiveP++;
            }
            return dec;
        }

        /**
         * @brief Initializes the network state for a new path traversal. 
         * 
         * @details
         * Prepares the network for sequential decision-making by:
         * 1. Clearing all node usage flags
         * 2. Resetting traverse counters for all nodes and the network
         * 3. Setting the current node to the start node's target
         * 4. Resetting fitness, validity, and consecutive processing node counters
         * 
         * After calling this method, the network is ready to receive observations
         * via decisionAndNextNode() one step at a time.
         *
         * @param startingFitness Optional initial fitness value to set before traversal (default is 0)
         */
        /**
         * @brief Aktualisiert die Lineage-Fitness-Statistik (inkrementeller Online-Mittelwert
         *        nach Welford) mit einer neuen Seed-Beobachtung dieser Linie.
         *
         * @details Gibt jeder Beobachtung gleiches Gewicht (im Gegensatz zu EMA, das jüngere
         * Beobachtungen stärker gewichtet). Dadurch entsteht ein echter Mittelwert über ALLE
         * jemals von dieser Linie gesehenen Seeds, unabhängig davon, wie lange sie zurückliegen.
         * Rein additiv: beeinflusst standardmäßig keine bestehende Selektion.
         *
         * @param rawFitness Rohreward-Fitness der aktuellen Seed-Episode.
         * @param landed true, falls diese Episode erfolgreich gelandet ist (rawFitness > LANDING_SUCCESS_THRESHOLD).
         */
        void updateLineageStats(float rawFitness, bool landed) {
            lineageN++;
            float delta = rawFitness - lineageMean;
            lineageMean += delta / static_cast<float>(lineageN);
            float delta2 = rawFitness - lineageMean;
            lineageM2 += delta * delta2;

            lineageSuccessN++;
            float landedValue = landed ? 1.0f : 0.0f;
            lineageSuccessRate += (landedValue - lineageSuccessRate) / static_cast<float>(lineageSuccessN);
        }

        /**
         * @brief Liefert die (unverzerrte) Stichproben-Varianz von lineageMean, basierend auf
         *        dem Welford-M2-Akkumulator lineageM2.
         *
         * @return 0.0f falls lineageN <= 1 (Varianz nicht definierbar), sonst lineageM2/(lineageN-1).
         *         Wird mit max(..., 0.0f) gegen negative Rundungsfehler abgesichert.
         */
        float lineageVariance() const {
            if (lineageN <= 1) return 0.0f;
            return std::max(0.0f, lineageM2 / static_cast<float>(lineageN - 1));
        }

        /**
         * @brief Liefert eine untere Konfidenzgrenze (Lower Confidence Bound, LCB) fuer
         *        lineageMean: lineageMean - z * Standardfehler, mit Standardfehler =
         *        sqrt(lineageVariance() / lineageN).
         *
         * @details Bestraft Linien mit wenigen Beobachtungen (kleines lineageN, hohe
         * Varianz der Schaetzung) automatisch mit einem groesseren Abschlag -- verhindert,
         * dass frisch ueber eine Schwelle (z.B. minLineageN) gekommene "Gluecks-Neulinge"
         * durch reines Stichprobenrauschen einen etablierten, praeziser geschaetzten
         * Champion bei der Elite-/Turnierauswahl verdraengen (Winner's-Curse-Problem).
         *
         * @param z Konfidenz-Multiplikator (z.B. 1.0 ≈ 84%, 1.645 ≈ 95%, 2.0 ≈ 97.7%
         *          einseitiges Konfidenzniveau bei Normalverteilungsannahme). Groesseres
         *          z = konservativerer (staerker abgestrafter) Schaetzwert.
         * @return lineageMean fuer lineageN <= 1 (Standardfehler nicht definierbar),
         *         sonst lineageMean - z * sqrt(lineageVariance()/lineageN).
         */
        float lineageLCB(float z) const {
            if (lineageN <= 1) return lineageMean;
            float standardError = std::sqrt(lineageVariance() / static_cast<float>(lineageN));
            return lineageMean - z * standardError;
        }

        /**
         * @brief Vermischt die Lineage-Fitness-Statistik von "other" in diese Instanz hinein
         *        (verwendet bei Crossover, wenn genetisches Material von "other" übernommen wird).
         *
         * @details Bildet einen beobachtungsgewichteten gepoolten Mittelwert aus den beiden
         * laufenden Mittelwerten (this und other). Um zu verhindern, dass eine sehr alte,
         * hochgezählte Linie neue Evidenz erdrückt, wird die effektive Beobachtungszahl beider
         * Seiten vor der Poolbildung auf priorCap gedeckelt. Nach dem Aufruf enthält "this"
         * die vermischte Statistik; "other" bleibt unverändert (einseitige Operation - bei
         * beidseitigem Genfluss muss die Methode für beide Individuen aufgerufen werden).
         *
         * @param other Das andere am Crossover beteiligte Individuum, dessen Statistik eingemischt wird.
         * @param priorCap Obergrenze für die bei der Poolbildung berücksichtigte Beobachtungszahl jeder Seite.
         */
        void blendLineageWith(const Network& other, int priorCap) {
            int nThis = std::min(lineageN, priorCap);
            int nOther = std::min(other.lineageN, priorCap);
            int nTotal = nThis + nOther;
            if (nTotal > 0) {
                // M2 wird vor dem Poolen auf die gedeckelte Beobachtungszahl umskaliert:
                // wir nehmen die aktuelle (unkorrigierte) Varianzschaetzung als "wahre"
                // Streuung an und rekonstruieren daraus ein M2, das zur gedeckelten
                // Stichprobengroesse passt (M2 = Varianz * (n_capped - 1)). So bleibt die
                // Varianzschaetzung bei einer Deckelung konsistent mit lineageN/lineageMean.
                float m2ThisCapped = (nThis > 1) ? lineageVariance() * static_cast<float>(nThis - 1) : 0.0f;
                float m2OtherCapped = (nOther > 1) ? other.lineageVariance() * static_cast<float>(nOther - 1) : 0.0f;

                float delta = other.lineageMean - lineageMean; // vor dem Ueberschreiben von lineageMean berechnen
                lineageM2 = m2ThisCapped + m2OtherCapped +
                            delta * delta * static_cast<float>(nThis) * static_cast<float>(nOther) / static_cast<float>(nTotal);

                lineageMean = (static_cast<float>(nThis) * lineageMean + static_cast<float>(nOther) * other.lineageMean) / static_cast<float>(nTotal);
                lineageN = std::min(lineageN, priorCap) + std::min(other.lineageN, priorCap);
            }

            int nThisS = std::min(lineageSuccessN, priorCap);
            int nOtherS = std::min(other.lineageSuccessN, priorCap);
            int nTotalS = nThisS + nOtherS;
            if (nTotalS > 0) {
                lineageSuccessRate = (static_cast<float>(nThisS) * lineageSuccessRate + static_cast<float>(nOtherS) * other.lineageSuccessRate) / static_cast<float>(nTotalS);
                lineageSuccessN = std::min(lineageSuccessN, priorCap) + std::min(other.lineageSuccessN, priorCap);
            }
        }

        /**
         * @brief Deckelt die Lineage-Beobachtungszahl auf priorCap (analog zum Crossover-
         *        Cap), damit auch reine Mutations-Linien ihre alte Historie "vergessen"
         *        koennen und neue Beobachtungen nach einer genetischen Veraenderung schneller
         *        ins Gewicht fallen. Rein monoton (min), daher gefahrlos mehrfach pro
         *        Generation aufrufbar (z.B. nach mehreren Mutationsoperatoren).
         *
         * @param priorCap Obergrenze fuer lineageN/lineageSuccessN nach dieser Operation.
         */
        void capLineageStats(int priorCap) {
            if (lineageN > priorCap) {
                // lineageM2 konsistent mit der neuen, kleineren Stichprobengroesse
                // umskalieren (siehe blendLineageWith() fuer dieselbe Logik): die
                // Varianzschaetzung bleibt erhalten, nur die "Konfidenz" (n) sinkt.
                float variance = lineageVariance();
                int cappedN = priorCap;
                lineageM2 = (cappedN > 1) ? variance * static_cast<float>(cappedN - 1) : 0.0f;
            }
            lineageN = std::min(lineageN, priorCap);
            lineageSuccessN = std::min(lineageSuccessN, priorCap);
        }

        void initPathTraversal(double startingFitness = 0){
            clearUsedNodes();
            for(auto& node : innerNodes){
                node.traverseCounter = 0;
                node.lastVisitStep = 0;
            }
            traverseCounter = 0;
            currentNodeID = startNode.edges[0];
            innerNodes[currentNodeID].used = true;
            innerNodes[currentNodeID].traverseCounter += 1;
            traverseCounter++;
            innerNodes[currentNodeID].lastVisitStep = traverseCounter;
            fitness = startingFitness;
            nConsecutiveP = 0;
            invalid = false;
        }

                /**
         * @brief Backward return pass: updates EdgeExperience for all JE nodes after an episode.
         *
         * @details
         * Iterates through episodeLog in reverse, computing the per-node discounted
         * return G_t = r_t + gamma_i * G_{t+1} independently for each JE node
         * (each node uses its own evolvable gamma).
         *
         * Complexity: O(T * |JE nodes|), where T = episode length.
         * The episodeLog is cleared after the update.
         *
         * Credit assignment note: G_t naturally attributes positive downstream
         * rewards (e.g., soft landing) back to earlier costly actions (e.g.,
         * fuel burn), weighted by gamma. A node with gamma ≈ 1 is long-sighted;
         * gamma ≈ 0 is purely myopic.
         */
        void updateExperienceFromEpisode() {
            if (episodeLog.empty()) return;

            std::vector<int> jeIDs;
            jeIDs.reserve(innerNodes.size());
            for (int i = 0; i < static_cast<int>(innerNodes.size()); i++) {
                if (innerNodes[i].type == "JE") jeIDs.push_back(i);
            }
            if (jeIDs.empty()) {
                episodeLog.clear();
                return;
            }

            const int T = static_cast<int>(episodeLog.size());  // ← Gesamtlänge
            std::vector<float> G(innerNodes.size(), 0.0f);

            for (int t = T - 1; t >= 0; t--) {
                float r = episodeLog[t].reward;

                for (int id : jeIDs) {
                    G[id] = r + innerNodes[id].gamma * G[id];
                }

                const int remainingSteps = T - t;  // ← Steps ab t bis Episodenende

                for (const auto& visit : episodeLog[t].jeVisits) {
                    innerNodes[visit.nodeID].updateEdgeExperience(
                        visit.edgeIndex,
                        G[visit.nodeID],
                        visit.obsF,
                        remainingSteps
                    );
                }
            }

            episodeLog.clear();
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
            innerNodes[currentNodeID].traverseCounter += 1;
            innerNodes[currentNodeID].lastVisitStep = traverseCounter;
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
         * @param gamma discount factor of the rewards
         * @param newRun If true, resets network state for a new episode; if false, continues from current state (useful for multi-episode evaluation)
         * @param validation If true, does not apply worstFitness penalty when constraints are violated (useful for validation runs where we want to observe rewards without penalization)
         * @param updateExperience If true, updates the experience of JE nodes after the episode (calls updateExperienceFromEpisode())
         * @param curriculumLevel Float between 0.0 and 1.0 controlling the difficulty of the environment (if supported).
         * @param absoluteImpulseCurriculum If true, curriculumLevel controls the ABSOLUTE strength of the
         *        initial random impulse (only its direction stays seed-random) so every seed trains at the same
         *        push strength at a given curriculum stage. If false (default), curriculumLevel scales each seed's
         *        own (seed-random) impulse relatively, preserving the natural difficulty ordering between seeds;
         *        at curriculumLevel 1.0 the environment stays exactly as Gymnasium produced it.
         * @param uniformDirectionCurriculum If true, the impulse DIRECTION is also forced to directionAngle
         *        (radians) instead of the seed's natural direction -- used to guarantee an evenly spread set of
         *        directions across a batch of seeds, rather than leaving direction to (non-uniform) chance.
         * @param directionAngle Forced impulse direction in radians, only used if uniformDirectionCurriculum=true.
         * @param survivalMode If true, Gymnasium's built-in reward is ignored entirely; fitness = number of
         *        steps survived (see SURVIVAL MODE note below). Takes precedence over potential if both are true.
         * @param potential If true (and survivalMode=false), uses potential-based reward shaping (PBRS) on top
         *        of Gymnasium's raw reward: r' = r + gamma*phi(s') - phi(s) with gamma = 1 and phi(terminal) = 0,
         *        where phi = -(|angle| + |angularVelocity| + |vy|). Because the shaping telescopes to -phi(s_0),
         *        the episode return shifts by a constant that depends on the seed only -- the ranking of
         *        individuals on any given seed is provably unchanged. If both survivalMode and potential are
         *        false (default), the raw, unmodified Gymnasium reward is used ("standard" mode).
         * 
         * @note SURVIVAL MODE: this function intentionally ignores Gymnasium's built-in reward
         *       (distance/velocity/angle shaping, fuel cost, landing bonus, crash penalty) entirely.
         *       Fitness is simply the number of steps survived (+1 per env.step()). Since episode
         *       termination -- whether caused by a crash, by a successful landing (Gymnasium ends the
         *       episode once the lander is "asleep"/at rest), or by leaving the frame -- always stops
         *       the reward accumulation, both crashing AND landing are implicitly disfavoured equally:
         *       the network is only rewarded for staying alive/airborne inside the frame as long as
         *       possible (up to maxSteps), with no notion of "reaching the pad" at all. This is a
         *       deliberate ablation to test whether the GNP can learn pure hover/balance control.
         * 
         * @warning The network must produce valid actions for the specific Gymnasium environment
         */
        void fitGymnasium(
            GymEnvWrapper& env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            int seed,
            bool newRun = true,
            bool validation = false,
            bool updateExperience = true,
            float curriculumLevel = 1.0f,
            bool absoluteImpulseCurriculum = false,
            bool uniformDirectionCurriculum = false,
            float directionAngle = 0.0f,
            bool survivalMode = false,
            bool potential = false
            ){

            // Curriculum-Erzwingung wird bewusst NUR waehrend des Trainings angewendet;
            // Validierungslaeufe (validation=true) nutzen immer die echte, ungedeckelte
            // Umgebung (siehe GymEnvWrapper::reset()).
            auto reset_out = env.reset(seed=seed, curriculumLevel, absoluteImpulseCurriculum, validation, uniformDirectionCurriculum, directionAngle);// Initial observation for the episode
            auto obs = reset_out[0].cast<std::vector<double>>();   
            std::vector<double> prevObs = obs;  // vorheriger Step

            if(newRun == true){
                clearUsedNodes();
                // clearing traverseCounter for each node and network
                for(auto& node : innerNodes){
                    node.traverseCounter = 0;
                    node.lastVisitStep = 0;
                }
                traverseCounter = 0;
            }

            nConsecutiveP = 0;
            currentNodeID = startNode.edges[0];
            innerNodes[currentNodeID].used = true;
            innerNodes[currentNodeID].traverseCounter += 1;
            traverseCounter ++;
            innerNodes[currentNodeID].lastVisitStep = traverseCounter;
            int dec;
            fitness = 0;
            lastFitness = 10;
            lastFitnessII = 2.5;
            nConsecutiveP = 0;
            invalid = false;
            bool done = false;
            int steps = 0;
            bool hasLanded = false;
            int decBefore = 0;

            int curriculumCutoff = maxSteps;
            // Discount of the potential-based shaping below. MUST stay 1.0: the fitness is
            // an UNDISCOUNTED sum over the episode, and only with gamma = 1 does the
            // shaping telescope to a constant offset per seed (see the potential branch).
            constexpr float shapingGamma = 1.0f;
            std::vector<float> rewards;
            rewards.reserve(maxSteps);

            // survivalMode / potential are now function parameters (see docstring above),
            // instead of hardcoded local flags -- lets callers select the reward mode
            // per call while keeping gymnasium()/gymnasiumMultiSeed() consistent.

            float angle_init = static_cast<float>(obs[4]);
            float angular_vel_init = static_cast<float>(obs[5]);
            float vy_init = static_cast<float>(obs[3]);

            float phi_t = -(std::abs(angle_init) + std::abs(angular_vel_init) + std::abs(vy_init));

            // --- Epsilon-Lexicase: written in EVERY reward mode (survivalMode/potential/raw),
            // see Network::lastEpisodeObjectives.
            bool landedThisEpisode = false;
            // Raw Gymnasium reward, summed independently of the reward mode chosen below,
            // so objective [0] stays comparable no matter how fitness is computed.
            float objGymRewardSum = 0.0f;

            while(done == false){

                // Observation + Delta konkatenieren
                // std::vector<double> obsWithDelta = obs;
                // for (size_t i = 0; i < obs.size(); ++i)
                //     obsWithDelta.push_back(obs[i] - prevObs[i]);

                //dec = decisionAndNextNode(obsWithDelta, dMax);
                dec = decisionAndNextNode(obs, dMax);

                if(dec == 4){
                    dec = decBefore;
                } else{
                    decBefore = dec;
                }

                if (invalid || nConsecutiveP > maxConsecutiveP){

                    //if(validation == false){
                    //    fitness += worstFitness;
                    //}                     

                    //episodeLog.clear();
                    //return;
                    dec = 0;
                }

                auto result = env.step(dec);
                prevObs = obs;
                obs = result[0].cast<std::vector<double>>(); 

                // --- Epsilon-Lexicase objectives, accumulated in EVERY reward mode ----
                // The landing flag is read off Gymnasium's raw reward (+100 on a safe
                // landing), so it stays correct no matter which reward branch runs below.
                {
                    float rawStepReward = result[1].cast<float>();
                    objGymRewardSum += rawStepReward;
                    if(rawStepReward >= 90.0f) landedThisEpisode = true;
                }

                // indcrease or decsrese landing reward
                // if (result[1].cast<float>() > 80.0f) {
                //     fitness -= 95.0f; // Strafpunkt fuer Landung ausserhalb des erlaubten Bereichs
                // } else if (result[1].cast<float>() <= -80.0f) {
                //     fitness += 95.0f; // Bonus fuer Landung innerhalb des erlaubten Bereichs
                // }
                //

                // Vertikaler Rahmen-Check: Gymnasium selbst terminiert nur bei
                // horizontalem Verlassen des Frames (abs(x) >= 1.0) oder bei Crash/
                // Landung -- die Box2D-Simulation laeuft aber UNBEGRENZT weiter, wenn
                // der Lander nach OBEN aus dem sichtbaren Bild hinausfliegt. Ohne
                // diesen Check kann eine Policy durch reinen Dauer-Hauptschub nach oben
                // aus dem Bild fliegen und so kuenstlich "ueberleben", ohne je zu landen
                // oder abzustuerzen (beobachtetes Exploit-Verhalten). frameHeightCap
                // entspricht (aus der obs-Skalierung in lunar_lander.py hergeleitet:
                // obs[1] = (pos.y - helipad_y - LEG_DOWN/SCALE) / (VIEWPORT_H/SCALE/2))
                // in etwa der Hoehe des oberen Bildschirmrands -- der Lander startet
                // dort bereits bei ca. 1.4. Ueberschreiten bedeutet: der Lander hat den
                // sichtbaren Rahmen nach oben verlassen.
                if (survivalMode) {

                    constexpr float frameHeightCap = 2.0f;
                    bool outOfFrameTop = static_cast<float>(obs[1]) > frameHeightCap;

                    if (outOfFrameTop || obs[6] > 0.5 || obs[7] > 0.5) {
                        done = true;
                    }
     
                    constexpr float survivalStepReward = 1.0f;
                    float stepReward = survivalStepReward;
                    fitness += survivalStepReward;
                    steps ++;
                    episodeLog.push_back({currentStepJEVisits, stepReward});
                    currentStepJEVisits.clear(); 

                    float r = survivalStepReward;
                    rewards.push_back(r);

                } else if (potential){

                    // Potential-based reward shaping in its exact form (Ng, Harada &
                    // Russell 1999): r'(s,s') = r(s,s') + gamma*Phi(s') - Phi(s), with
                    // Phi(terminal) = 0. At gamma = 1 the shaping telescopes over the
                    // episode to -Phi(s_0) -- a constant that depends on the seed only, not
                    // on the individual. The episode return therefore keeps every ranking
                    // intact; the shaping cannot bias selection (see
                    // tests/test_potential_shaping.py).
                    //
                    // Truncation at maxSteps is treated as terminal on purpose. Leaving
                    // Phi(s_last) in the sum would make the offset depend on where the
                    // individual happened to stop and would break exactly that guarantee.
                    float gymReward = result[1].cast<float>();

                    bool episodeEnds = result[2].cast<bool>() || result[3].cast<bool>()
                                       || (steps + 1) >= maxSteps;

                    float phi_next = 0.0f;
                    if(!episodeEnds){
                        phi_next = -(std::abs(static_cast<float>(obs[4]))
                                   + std::abs(static_cast<float>(obs[5]))
                                   + std::abs(static_cast<float>(obs[3])));
                    }

                    float stepReward = gymReward + shapingGamma * phi_next - phi_t;
                    phi_t = phi_next;

                    fitness += stepReward;
                    steps ++;
                    episodeLog.push_back({currentStepJEVisits, stepReward});
                    currentStepJEVisits.clear(); 

                    rewards.push_back(stepReward);

                } else {

                    float stepReward = result[1].cast<float>();
                    steps ++;
                    episodeLog.push_back({currentStepJEVisits, stepReward});
                    currentStepJEVisits.clear(); 

                    rewards.push_back(stepReward);
                    fitness += stepReward;

                    bool leftContact  = obs[6] > 0.5;
                    bool rightContact = obs[7] > 0.5;
                    if ((leftContact || rightContact) && !hasLanded) {
                        float vx = static_cast<float>(obs[2]);
                        float vy = static_cast<float>(obs[3]);
                        lastFitness = std::sqrt(vx * vx + vy * vy); // Betrag der Landegeschwindigkeit (nur Diagnose)
                        float x = static_cast<float>(obs[0]);
                        lastFitnessII = std::abs(x); 
                        hasLanded = true;
                    }

                }

                // float vx = static_cast<float>(obs[2]);
                // float vy = static_cast<float>(obs[3]);
                // lastFitness += std::abs(std::sqrt(vx * vx + vy * vy)); // Betrag der Geschwindigkeit 

                if(result[2].cast<bool>() || result[3].cast<bool>() || steps >= maxSteps) done = true; 
                // if(result[2].cast<bool>() || result[3].cast<bool>()) done = true; 

                // if(done == true && result[1].cast<float>() > 95){
                // //    lastFitness = 10; 
                // //    lastFitnessII = 2.5;
                //     float x = static_cast<float>(obs[0]);
                //     if (x < -0.2f || x > 0.2f) {
                //         fitness -= 100.0f; // Strafpunkt für Landung außerhalb des erlaubten Bereichs
                //     }
                // }
                //
                float x = static_cast<float>(obs[0]);
                lastFitnessII = std::abs(x); 
            }

            // --- Epsilon-Lexicase objectives, see Network::lastEpisodeObjectives for why
            // exactly these four. All are read off the final frame except the step count.
            {
                float vx_final = static_cast<float>(obs[2]);
                float vy_final = static_cast<float>(obs[3]);
                float x_final  = static_cast<float>(obs[0]);
                // Penalties just beyond the worst value each quantity actually reaches
                // (speeds stay below ~3, |x| below ~1.5 before the frame ends), so a
                // failed episode ranks below every landing without blowing up the spread
                // the epsilon tolerance is derived from.
                constexpr float NO_LANDING_SPEED = -4.0f;
                constexpr float NO_LANDING_POSITION = -2.0f;
                lastEpisodeObjectives[0] = objGymRewardSum;
                lastEpisodeObjectives[1] = landedThisEpisode ? -std::abs(vy_final) : NO_LANDING_SPEED;
                lastEpisodeObjectives[2] = landedThisEpisode ? -std::abs(vx_final) : NO_LANDING_SPEED;
                lastEpisodeObjectives[3] = landedThisEpisode ? -std::abs(x_final) : NO_LANDING_POSITION;
                lastEpisodeObjectives[4] = landedThisEpisode ? 1.0f : 0.0f;
            }

            // 3) Nach der Episode: Return-to-Go rückwärts + Cutoff-Aggregation
            // std::vector<float> G(rewards.size(), 0.0f);
            // float running = 0.0f;
            // for (int t = static_cast<int>(rewards.size()) - 1; t >= 0; --t) {
            //     running = rewards[t] + gamma * running;
            //     G[t] = running;
            // }
            //
            // int T = static_cast<int>(G.size());
            // int cut = std::min(curriculumCutoff, T);   // oder: (curriculumCutoff < 0 ? T : std::min(curriculumCutoff, T))

            // float sum = 0.0f;
            // for (int t = 0; t < cut; ++t) sum += G[t];
            // if(cut > 0 && validation == false){
            //     // fitness = sum / static_cast<float>(cut);
            //     // fitness = sum;
            //     // fitness = sum / 1000;
            //     fitness = G[0];
            // } 
            //
            //if (frozenExperience)
              //  episodeLog.clear(); // Elite: Log immer leeren
            //else if (updateExperience)
              //  updateExperienceFromEpisode(); // Einzelaufruf: sofort updaten
            updateExperienceFromEpisode(); // Einzelaufruf: sofort updaten
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
            innerNodes[currentNodeID].traverseCounter += 1;
            innerNodes[currentNodeID].lastVisitStep = traverseCounter;
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
         * @brief Validates node indices and edge connections in the network.
         * 
         * @details
         * This method performs integrity checks on the network's structure by:
         * 1. Verifying that each node's ID matches its index in the innerNodes vector
         * 2. Ensuring that all edges point to valid node indices within the bounds of innerNodes
         * 
         * If any inconsistencies are found (e.g., node ID mismatch or edge pointing to non-existent node), 
         * error messages are printed to standard error output. 
         */
        void checkNodeIndicesAndEdges(std::string msg=""){
            for(int n=0; n<innerNodes.size();n++){
                if(innerNodes[n].id != n){
                    std::cerr << "[ERROR] after" << msg << "Node index mismatch: node at index " << n << " has id " << innerNodes[n].id << std::endl;
                }
                for(auto& edge : innerNodes[n].edges){
                    if(edge > innerNodes.size()-1){
                        std::cerr << "[ERROR] after" << msg << "Edge index out of bounds: node " << n << " has edge pointing to " << edge << " but max index is " << innerNodes.size()-1 << std::endl;
                    }
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
         * 4. If an edge points to the node itself (edge == node.id), creating a self-loop
         * 5. Calls the node's changeEdge() method to randomly select a new valid target node
         * 
         * This ensures the network graph remains well-defined with all edges pointing
         * to existing nodes and preventing self-loops, which helps avoid runtime errors 
         * during network traversal.
         * 
         * @note This function should be called after any operation that removes nodes
         */
        void changeFalseEdges(){
            for(auto& node : innerNodes){
                for(auto& edge : node.edges){
                    if(edge > innerNodes.size()-1 || edge == node.id){ // edge has no successor node or pointing to itself -> change edge to a random valid node
                        edge = node.changeEdge(innerNodes.size(), edge, &innerNodes);
                    }
                }
            }
        }

        /**
         * @brief Remaps node IDs and their associated edges using a provided mapping.
         * 
         * @details This method updates node IDs and edge references for a subset of nodes
         * specified by their indices. For each node in the given indices, if its ID exists
         * in the mapping, it is replaced with the mapped value. Similarly, all edges of
         * these nodes are updated if they exist in the mapping. This is useful for
         * renumbering or consolidating node identifiers while maintaining 
         * graph connectivity. 
         * 
         * @param map A reference to an unordered map where keys are old node IDs and values
         * are new node IDs to remap to.
         * @param nodeIndices A vector of indices specifying which nodes in the innerNodes
         * collection should be processed for remapping.
         * @param includeStartNode If true, remaps the first edge of the start node if it
         * exists in the mapping. Defaults to false.
         */
        void remapNodeIdsAndEdges(
                std::unordered_map<int, 
                int>& map, 
                const std::vector<int>& nodeIndices, 
                bool includeStartNode = false){

            if(map.empty()) {
                return;
            }

            if(includeStartNode){
                if(map.contains(startNode.edges[0])){
                    startNode.edges[0] = map[startNode.edges[0]];
                } 
            }

            for(int ni: nodeIndices){

                if(map.contains(innerNodes[ni].id)){
                    innerNodes[ni].id = map[innerNodes[ni].id];
                } 

                for(size_t edgeIdx = 0; edgeIdx < innerNodes[ni].edges.size(); ++edgeIdx){
                    auto& edge = innerNodes[ni].edges[edgeIdx];
                    if(map.contains(edge)){
                           edge = map[edge];
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
         * - ALL deletable unused nodes are removed per call -- deliberately asymmetric to the
         *   addition branch, which adds at most ONE node per call (break). The junk quota is the
         *   stopping condition: deletion continues while unused - 1 > size * junk, i.e. the network
         *   is trimmed down to the allowed share of junk DNA (with junk = 0, every unused node goes).
         *   Rationale: crossover(type="seedSpecialist") appends whole sub-graphs in one step, so a
         *   deletion capped at one node per generation can never balance the growth.
         * 
         * @param minF Vector of minimum feature values for each feature dimension (used for judgment node boundary initialization)
         * @param maxF Vector of maximum feature values for each feature dimension (used for judgment node boundary initialization)
         * @junk ratio of protected unused nodes (junk DNA). A value of 0.1 protects 10% of unused nodes.
         * 
         * @warning This method must be called bevore edgeMutation()! Reason: if edges are change 
         * by edgeMutation(), the node flag "used" is not guaranteed to be correct.
         *
         * @post All edges remain valid (no dangling edges)
         * @post Node IDs are contiguous from 0 to innerNodes.size()-1
         * 
         */
        bool addDelNodes(
                std::vector<float>& minF, 
                std::vector<float>& maxF, 
                float junk, 
                std::vector<int>& nFeatureValues,
                int currentGeneration,
                int crossoverProtection = 3,
                int nodeGracePeriod = -1
                ){
            bool changed = false;
            std::bernoulli_distribution distributionBernoulliAdd(0.5);
            std::bernoulli_distribution distributionBernoulliProcessingNode(pnRatio());
            bool resultAdd = distributionBernoulliAdd(*generator);
            countUsedNodes();

            // ------------------------------------------------------------------
            // Build meta-node candidate list for cluster-aware edge initialisation.
            // Only built when cluster information is available.
            // Candidates = {Entry(C_k) for each cluster k} + {unclustered node IDs}
            // ------------------------------------------------------------------
            std::vector<int> metaCandidates;

            //     for (auto& node : innerNodes)
            // }

            for(int n=0; n<innerNodes.size(); n++){
                if(resultAdd && // adding node
                    innerNodes.size() - nUsedNodes <= innerNodes.size() * junk){// left: current junk (unused nodes); right allowed junk 
                                                                                
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
                            innerNodes.back().setEdges("P", static_cast<int>(innerNodes.size()), 0, metaCandidates);
                            pn += 1;
                    }else{ // add judgment node
                        std::uniform_int_distribution<int> distributionJNF(0, jnf-1);
                        int randomInt = distributionJNF(*generator);
                        int nOutgoingEdges;
                        std::string jNodeType = useExperience ? "JE" : "J"; // NEU
                        innerNodes.push_back(Node(
                                    generator,
                                    innerNodes.size(),
                                    jNodeType,        // NEU: "J" oder "JE"
                                    randomInt
                                    ));

                        // Jede strukturelle Neuerung bekommt eine frische, global eindeutige
                        // innovationID -- das historische Kennzeichen, an dem crossover(type=
                        // "innovation") spaeter erkennt, welche Knoten zweier Individuen
                        // einander entsprechen (siehe Node::innovationID).

                        if(nFeatureValues.size() > 0){
                             nOutgoingEdges = nFeatureValues[randomInt];
                        } else {nOutgoingEdges = 0;}

                        if (jNodeType == "JE") {
                            // NEU: Initialize experience structures for JE node
                            innerNodes.back().initEdgeExperience();
                        }


                        if(fractalJudgment == false || nOutgoingEdges != 0){
                            innerNodes.back().setEdges("J", innerNodes.size(), nOutgoingEdges, metaCandidates);
                            innerNodes.back().setEdgesBoundaries(minF[randomInt], maxF[randomInt]);
                        }
                        else if(fractalJudgment == true && nOutgoingEdges == 0){
                            std::pair<int, int> k_d = random_k_d_combination(innerNodes.size()-1, generator);
                            innerNodes.back().k_d.first = k_d.first;
                            innerNodes.back().k_d.second = k_d.second;
                            innerNodes.back().setEdges("J", innerNodes.size(), pow(k_d.first,k_d.second), metaCandidates);
                            innerNodes.back().productionRuleParameter = randomParameterCuts(innerNodes.back().k_d.first-1, generator);
                            std::vector<float> fractals = fractalLengths(innerNodes.back().k_d.second, sortAndDistance(innerNodes.back().productionRuleParameter));
                            innerNodes.back().setEdgesBoundaries(minF[randomInt], maxF[randomInt], fractals);
                        }
                    }

                    changed = true;
                    break; // NOTE: just one node can be added with break statement!

                }else if(!resultAdd && 
                        innerNodes.size() - nUsedNodes - 1 > innerNodes.size() * junk && // left: current junk size (unused nodes); right: allowed junk size 
                        innerNodes[n].used == false &&
                        (innerNodes[n].generationReceived == -1 || currentGeneration - innerNodes[n].generationReceived >= crossoverProtection) &&
                        // Keep nodes that were recently part of the active sub-graph: one
                        // generation's seed panel must not kill a node the next panel needs.
                        // Nodes never traversed have everUsed == false and stay immediately
                        // deletable -- those are genuine junk.
                        !(innerNodes[n].everUsed && innerNodes[n].unusedSince < nodeGracePeriod)
                        )
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

                    // Zaehler des geloeschten Typs mitfuehren, sonst driften jn/pn (und damit
                    // pnRatio()) mit jeder Loeschung weiter von der Realitaet ab.
                    if(innerNodes[n].type == "P"){
                        if(pn > 0) pn -= 1;
                    } else if(innerNodes[n].type == "J" || innerNodes[n].type == "JE"){
                        if(jn > 0) jn -= 1;
                    }

                    innerNodes.erase(innerNodes.begin()+n);

                    // Nach dem erase() ist der nachfolgende Knoten auf Position n gerueckt.
                    // Ohne dieses n-- wuerde ihn das n++ der Schleife ueberspringen, sodass
                    // pro Aufruf nur jeder ZWEITE loeschbare Knoten entfernt wird. Mit dem
                    // Dekrement werden alle unbenutzten Knoten bis zur junk-Quote entfernt --
                    // das block-wertige Anhaengen (seedSpecialist) bekommt damit eine
                    // gleich schnelle Gegenkraft. Hinzugefuegt wird weiterhin nur EIN Knoten
                    // pro Aufruf (break im Add-Zweig).
                    n--;
                    changed = true;
                }
            }
            innerNodes.shrink_to_fit();
            return changed;
        }
        
        /**
         * @brief Counts the total number of edges in the network, optionally filtering by used nodes.
         * 
         * @details
         * This method iterates through all inner nodes and sums the number of outgoing edges. If the
         * justUsedNodes parameter is set to true, only nodes that are currently marked as used (used == true) are included in the count. 
         * 
         * @param justUsedNodes If true, only counts edges from nodes that are currently used. If false, counts edges from all nodes regardless of usage status.
         * @return The total count of edges in the network, filtered by usage if specified.
         */
        int countEdges(bool justUsedNodes = false, bool skipFrozenNodes = false){
            int count = 0;
            for(const auto& node : innerNodes){
                if((justUsedNodes && node.used == false) || (skipFrozenNodes && node.frozen > 0)){
                    continue; // skip unused nodes if justUsedNodes is true
                }
                count += node.edges.size();
            }
            return count;
        }

        /**
         * @brief Calculates the ratio of processing nodes to total nodes in the network.
         * 
         * @details
         * This method iterates through all inner nodes and counts the number of processing nodes (pn) and judgment nodes (jn). 
         * 
         * @return The ratio of processing nodes to total nodes, calculated as pn / (pn + jn). If there are no nodes, returns 0 to avoid division by zero.
         */
        float pnRatio(){
            // Neu zaehlen statt aufaddieren: zuvor wurden pn/jn bei JEDEM Aufruf um die
            // aktuellen Knotenzahlen ERHOEHT (kein Reset), sodass beide Zaehler mit jeder
            // Generation weiter anwuchsen und nicht mehr die Netzgroesse beschrieben --
            // das Verhaeltnis war dadurch ein traeger kumulativer Mittelwert statt der
            // aktuellen Zusammensetzung.
            unsigned int nP = 0, nJ = 0;
            for(auto& node : innerNodes){
                if(node.type == "P"){
                    nP += 1;
                } else if(node.type == "J" or node.type == "JE"){
                    nJ += 1;
                }
            }
            pn = nP;
            jn = nJ;
            if(nP + nJ == 0) return 0.0f;
            return static_cast<float>(nP) / static_cast<float>(nP+nJ);
        }










        

};
#endif
