#ifndef POPULATION_HPP
#define POPULATION_HPP
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <random>
#include <unordered_set>
#include <utility>
#include <cmath>
#include <thread>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <deque>
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
        /** Diagnostic: serialised state of the shared RNG, for comparing two runs. */
        std::string rngState() const {
            std::ostringstream ss;
            ss << *generator;
            return ss.str();
        }
        /** Diagnostic: identity of the shared RNG, to detect individuals pointing elsewhere. */
        std::uintptr_t rngPointer() const {
            return reinterpret_cast<std::uintptr_t>(generator.get());
        }

        /** @cond INTERNAL */
        const unsigned int ni; /**< Number of individuals in the population (constant after initialization) */
        unsigned int jn; /**< Initial number of judgment nodes per individual */
        unsigned int jnf; /**< Number of judgment node function types available */
        unsigned int pn; /**< Initial number of processing nodes per individual */
        unsigned int pnf; /**< Number of processing node function types available */
        bool fractalJudgment; /**< Flag indicating whether judgment nodes use fractal-based edge patterns */
        bool useExperience; ///< If true, all judgment nodes are created as "JE" nodes
        std::vector<Network> individuals; /**< Vector containing all Network individuals in the population */
        float bestFit; /**< Fitness value of the best individual in the current population */
        std::vector<int> indicesElite; /**< Indices of elite individuals (protected from mutation) */
        float meanFitness = 0; /**< Mean fitness across all individuals in the population */
        float minFitness; /**< Minimum fitness value in the current population */
        int maxNetworkSize; 
        std::vector<int> nFeatureValues; /** stores the number of feature values */
        // Crossover diagnostics of the LAST crossover() call. Only type="semantic" fills
        // them; all other types leave them at 0. They exist to tell "the operator had no
        // effect" apart from "the operator never fired": if nodesSkipped dominates
        // nodesExchanged, boundaryTolerance is too tight for a gene to travel with its
        // whole neighbourhood (see the all-or-nothing rule in the semantic branch).
        int crossoverPairsApplied = 0;   /**< Parent pairs that actually entered the exchange. */
        int crossoverNodesMatched = 0;   /**< Node pairs the role matching found, summed over all pairs. */
        int crossoverNodesExchanged = 0; /**< Nodes actually transferred donor -> recipient. */
        int crossoverNodesSkipped = 0;   /**< Nodes drawn for exchange but skipped: an edge was untranslatable. */
        int nextTransplantBlockID = 1; /**< Monotonically increasing counter used to tag each transplanted
                                             sub-graph produced by crossover(type="seedSpecialist") with a unique
                                             Node::transplantBlockID (0 is reserved / unused, -1 means "no block"). */
        /** @endcond */

        /** @name Self-adaptive mutation strength (Rechenberg 1/5-success-rule) */
        /** @{ */
        using SuccessCriterion = std::function<float(const Network&)>; /**< Exchangeable per-individual scoring function for updateAdaptiveK(). May be a continuous score in [0,1] (e.g. seedConsistencyRatio()) or a legacy bool-returning predicate (implicitly converted to 1.0f/0.0f, e.g. seedConsistentSuccess()). */

        float adaptiveK_ = 1.0f; /**< Current self-adapted k, used as the mutation-strength parameter (see updateAdaptiveK()). */
        std::deque<float> successRateHistory_; /**< Rolling window of per-generation (mean) scores -- smooths the signal over multiple generations (see windowSize in updateAdaptiveK()). Most recent at the back. */
        /** @} */

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
         * @param _useExperience If true, all judgment nodes are created as "JE" nodes that track experience on edges; if false, standard "J" nodes are used
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
                bool _useExperience = false,
                std::vector<int> _nFeatureValues = {}
                ):
            generator(std::make_shared<std::mt19937_64>(seed)),
            ni(_ni),
            jn(_jn),
            jnf(_jnf),
            pn(_pn),
            pnf(_pnf),
            fractalJudgment(_fractalJudgment),
            useExperience(_useExperience),
            nFeatureValues(_nFeatureValues)
        {
            for(int i=0; i<ni; i++){
                individuals.push_back(Network(
                    generator, jn, jnf, pn, pnf,
                    fractalJudgment,
                    useExperience,
                    nFeatureValues
                ));
            }
            // Die Startnetze belegen bereits die innovationIDs 0..jn+pn-1 (jedes Individuum
            // dieselben, siehe Network-Konstruktor). Der Zaehler fuer strukturell NEUE Knoten
            // muss deshalb dahinter beginnen.
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
                   if(node.type == "J" || node.type == "JE"){
                       if(fractalJudgment == true){
                           node.productionRuleParameter = randomParameterCuts(node.k_d.first-1, generator);
                           std::vector<float> fractals = fractalLengths(node.k_d.second, sortAndDistance(node.productionRuleParameter));
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f], fractals);
                       }else {
                           node.setEdgesBoundaries(minF[node.f], maxF[node.f]);
                       }
                       if(node.type == "JE") node.initEdgeExperience();
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
         * @param validation If true, does not apply worstFitness penalty when constraints are violated (useful for validation runs where we want to observe rewards without penalization)
         * @param curriculumLevel Float between 0.0 and 1.0 controlling the difficulty of the environment (if supported).
         * @param absoluteImpulseCurriculum See Network::fitGymnasium().
         * @param uniformDirectionCurriculum If true, forces the impulse direction to directionAngle (see
         *        Network::fitGymnasium()) -- lets renderVideos() visually reflect the same forced-direction
         *        training conditions currently in effect, instead of always falling back to the natural seed
         *        direction.
         * @param directionAngle Forced impulse direction in radians, only used if uniformDirectionCurriculum=true.
         * 
         * @param survivalMode See Network::fitGymnasium(). Passed through unchanged so
         *        renderVideos()/gymnasium() (video path) always matches the reward mode
         *        used during training via gymnasiumMultiSeed().
         * @param potential See Network::fitGymnasium().
         * 
         * @see Network::fitGymnasium()
         */
        void gymnasium(
            GymEnvWrapper& env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            int seed,
            bool validation = false,
            float curriculumLevel = 1.0f,
            bool absoluteImpulseCurriculum = false,
            bool uniformDirectionCurriculum = false,
            float directionAngle = 0.0f,
            bool survivalMode = false,
            bool potential = false
                ){

            bool updateExperience = false;
            for(auto& network : individuals){
                network.fitGymnasium(
                        env,
                        dMax,
                        maxSteps,
                        maxConsecutiveP,
                        worstFitness,
                        seed,
                        true,
                        validation,
                        updateExperience,
                        curriculumLevel,
                        absoluteImpulseCurriculum,
                        uniformDirectionCurriculum,
                        directionAngle,
                        survivalMode,
                        potential
                        );
                // Konsistenz mit gymnasiumMultiSeed(): auch der Einzel-Seed-Pfad (Video-
                // Rendering) befuellt lexicaseObjectives, hier ohne Mittelung ueber
                // mehrere Seeds (nur 1 Episode).
                network.lexicaseObjectives = network.lastEpisodeObjectives;
                network.objectivesPerSeed.assign(1, network.lastEpisodeObjectives);
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
         * @param useLineageFitness Wenn true, wird als Selektionskriterium bevorzugt
         *        `lineageMean` (laufender Mittelwert ueber ALLE je gesehenen Seeds dieser
         *        Linie) statt der rohen aktuellen Fitness verwendet -- macht die Selektion
         *        robuster gegen Generalisierungs-Ausreisser (z.B. ein einzelner harter Crash-
         *        Seed in der aktuellen Generation). Gilt NUR fuer Individuen mit
         *        `lineageN >= minLineageN` (Bootstrap-Schutz); alle anderen (z.B. junge
         *        Linien direkt nach Populationsinitialisierung) fallen automatisch auf die
         *        rohe `fitness` zurueck, bis genug Beobachtungen vorliegen. Reporting-Groessen
         *        (bestFit/meanFitness/minFitness) bleiben unabhaengig davon immer die rohe
         *        Fitness des jeweiligen Turniersiegers (fuer konsistentes Logging/Plotting).
         * @param minLineageN Minimale Anzahl an Lineage-Beobachtungen, ab der `lineageMean`
         *        statt roher Fitness als Kriterium genutzt wird (nur relevant, wenn
         *        `useLineageFitness == true`). Default 10.
         * 
         * @note N ≥ 2 for meaningful selection pressure
         * @note Population size remains constant at ni
         * 
         */
        void tournamentSelection(int N, int E, bool useLineageFitness = false, int minLineageN = 10, float lineageZ = 0.0f){
            // Liefert das tatsaechliche Selektionskriterium fuer ein Individuum: entweder die
            // rohe aktuelle Fitness (Standard) oder -- falls aktiviert und genug Beobachtungen
            // vorliegen -- die untere Konfidenzgrenze (LCB) des Lineage-Mittelwerts. Die LCB
            // bestraft Linien mit kleinem lineageN/hoher Unsicherheit automatisch (siehe
            // Network::lineageLCB()) und verhindert so, dass frisch ueber minLineageN
            // gekommene "Gluecks-Neulinge" durch reines Stichprobenrauschen etablierte,
            // praeziser geschaetzte Linien bei der Selektion verdraengen (Winner's-Curse-Fix).
            // lineageZ=0.0 (Default) reduziert lineageLCB() auf den reinen lineageMean
            // (bisheriges Verhalten, abwaertskompatibel).
            auto selectionCriterion = [&](const Network& ind) -> float {
                if(useLineageFitness && ind.lineageN >= minLineageN){
                    return ind.lineageLCB(lineageZ);
                }
                return ind.fitness;
            };

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
                float bestCriterionTournament = std::numeric_limits<float>::lowest();
                int indexBestIndTournament = 0;
                tournament.clear();

                while(tournament.size()<N){ // set the tournament
                    int randomInt = distribution(*generator);
                    tournament.insert(randomInt);
                }
                for(int k : tournament){
                   float criterion = selectionCriterion(individuals[k]);
                   if(criterion > bestCriterionTournament){
                       bestCriterionTournament = criterion;
                       indexBestIndTournament = k;
                   } 
                }
                selection.push_back(individuals[indexBestIndTournament]);
                // Reporting bleibt konsistent auf roher Fitness, unabhaengig vom
                // verwendeten Selektionskriterium (siehe Doku oben).
                float bestFitTournament = individuals[indexBestIndTournament].fitness;
                meanFitness += bestFitTournament;
                if (bestFitTournament < minFitness) {
                    minFitness = bestFitTournament;
                }
                if (bestFitTournament > bestFit) {
                    bestFit = bestFitTournament;
                }
            }
            setElite(E, individuals, selection, useLineageFitness, minLineageN, lineageZ);
            individuals = std::move(selection);
            // set frozenExperience flag back for non-elite
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
            meanFitness /= individuals.size();
        }

        /**
         * @brief Performs Epsilon-Lexicase Selection, either over 5 isolated fitness
         *        objectives ("objectives") or over per-seed test cases with a
         *        3-tier hierarchy ("seeds").
         *
         * @details
         * Unlike tournamentSelection(), which compares individuals using a single
         * aggregated fitness scalar, lexicaseSelection() selects individuals by
         * filtering the candidate pool test case by test case, in a randomly
         * shuffled order drawn independently for every selection event.
         *
         * See: Spector, "Assessment of Problem Modality by Differential Performance
         * of Lexicase Selection in Genetic Programming", and La Cava et al.,
         * "Epsilon-Lexicase Selection for Regression".
         *
         * **type = "objectives" (default)**: test cases are the isolated objectives of
         * Network::lexicaseObjectives, each averaged over the seed batch. Five criteria,
         * higher = better, and EVERY one of them is maximised by landing on the pad (see
         * Network::lastEpisodeObjectives):
         *   1. **Total Gymnasium reward** over the episode.
         *   2. **Descent safety**: -|vy_end|, but only if the episode ended in a landing;
         *      otherwise a fixed penalty.
         *   3. **Lateral speed**: -|vx_end| under the same condition, kept separate from
         *      the vertical one -- a combined term lets a network score on the vertical
         *      part alone and slide off the pad sideways, which is what was measured.
         *   4. **Horizontal precision**: -|x_end|, again only when landed.
         *   5. **Landed**: 1.0 on a safe landing, 0.0 otherwise.
         * The condition on 2-4 matters: measured without it, a network hovering over the
         * pad until the step limit won four of the five test cases (it stands still, above
         * the pad, and survives longest), and the population drifted there.
         * Per selection event: shuffle the objective indices, then for each objective
         * (while >1 candidate remains) find the best value among the remaining
         * candidates, eliminate everyone below best-epsilon, continue to the next
         * objective. Random tie-break if >1 candidate remains at the end.
         *
         * **type = "objectivesPerSeed"**: test cases are the (seed x objective) pairs --
         * the 5 objectives above, kept separate for every seed
         * (network.objectivesPerSeed[seedIdx][objIdx]), giving 5*nSeeds test cases instead
         * of 5 or nSeeds. Motivation: with one scalar per seed, individuals on a converged
         * plateau become indistinguishable (their total rewards lie closer together than
         * the seed-to-seed noise), while the objectives still measure different things and
         * separate them. The whole grid is shuffled as one, so neither seeds nor objectives
         * are systematically decided first. Requires gymnasiumMultiSeed() to have run.
         *
         * **type = "seedsStandardized"**: same test cases as "seeds", but the rewards of each
         * seed are z-transformed over the whole population once per generation, and the
         * tolerance is a fixed number of standard deviations (`epsilon`, negative defaults to
         * 0.5) instead of the MAD automatic. This makes the selection pressure independent of
         * the shape and the scale of the reward and removes the seed difficulty, while --
         * unlike a rank transform -- keeping the magnitudes that epsilon is meant to judge.
         * Standardising over the population rather than over the shrinking candidate pool is
         * deliberate, so a fixed epsilon does not get more permissive along the filter chain.
         *
         * **type = "seedsRank"**: same test cases as "seeds", but each filter cuts by RANK
         * instead of by reward value: the remaining candidates are sorted by their reward on
         * that seed and the best (epsilon + 1) ranks are kept, ties surviving together. The
         * ranks are recomputed over the remaining pool at every step, so the selection
         * pressure stays constant along the chain. `epsilon` is therefore a number of rank
         * positions, not a reward margin, and the MAD automatic is not available (the MAD of
         * ranks is ~n/4 whatever the data); a negative epsilon defaults to 3. Motivation: the
         * reward is bimodal, so a value-based MAD mostly measures the distance between
         * "landed" and "crashed" rather than the resolution among near-equal candidates.
         * WARNING: an absolute rank cut collapses the pool to epsilon+1 candidates in a
         * single step, after which every remaining test case filters nobody -- only the
         * first one or two seeds ever decide a parent. Prefer "seedsStandardized".
         *
         * **type = "seeds"**: test cases are the individual seeds evaluated by
         * gymnasiumMultiSeed() (network.fitnessValues[seedIdx], the raw/aggregated
         * per-seed reward). For each seed in the shuffled order, the raw reward is
         * maximized directly (MAD- or fixed-epsilon tolerance) -- no landing/
         * landing-speed tiers, a single criterion per seed.
         *
         * **Algorithm per selection event** (both variants):
         * 1. Start with the full population as candidate pool.
         * 2. Draw a random permutation of test cases (objectives or seeds).
         * 3. Filter the candidate pool test case by test case as described above.
         * 4. If exactly 1 candidate remains, it is the selected parent. If more
         *    than 1 remains after all test cases are exhausted (tie), pick
         *    uniformly at random among the remaining candidates.
         * 5. For the next parent, restart from step 1 -- including a freshly
         *    drawn random shuffle of the test cases.
         *
         * @param E Elite size (number of best individuals to preserve unchanged, by aggregated fitness).
         * @param epsilon Tolerance for the per-test-case filtering step. If negative
         *                (default), an automatic MAD-based epsilon is computed per
         *                test case/round (standard Epsilon-Lexicase). If >= 0, this
         *                fixed value is used as the tolerance for every test case/round
         *                instead.
         * @param type One of "objectives" (default), "seeds", "seedsStandardized", "seedsRank"
         *             or "objectivesPerSeed" -- selects which test
         *             case decomposition to use (see above). Throws std::invalid_argument
         *             for any other value.
         *
         * @pre For type="objectives": all individuals must have a non-empty, equally-sized
         *      (5-D) lexicaseObjectives vector (populated by gymnasiumMultiSeed()/gymnasium()).
         * @pre For type="seeds": all individuals must have a non-empty and equally-sized
         *      fitnessValues vector (populated by gymnasiumMultiSeed()).
         *
         * @note Population size remains constant at ni.
         * @note Elite handling and bestFit/meanFitness/minFitness bookkeeping mirror
         *       tournamentSelection() for compatibility with the rest of the pipeline.
         */
        void lexicaseSelection(int E, float epsilon = -1.0f, std::string type = "objectives"){
            if(type != "objectives" && type != "seeds" && type != "objectivesPerSeed"
               && type != "seedsRank" && type != "seedsStandardized"){
                throw std::invalid_argument("lexicaseSelection: unknown type \"" + type + "\" (expected \"objectives\", \"seeds\", \"seedsStandardized\", \"seedsRank\" or \"objectivesPerSeed\")");
            }

            // Berechnet die MAD-basierte Toleranz (oder den festen epsilon-Wert) fuer eine
            // Menge von Werten -- identisch fuer jedes Ziel/jeden Seed/jede Runde benoetigt.
            auto computeThreshold = [&](const std::vector<float>& values) -> float {
                if(epsilon >= 0.0f) return epsilon;
                std::vector<float> sortedValues = values;
                std::sort(sortedValues.begin(), sortedValues.end());
                size_t mid = sortedValues.size() / 2;
                float median = sortedValues[mid];
                if(sortedValues.size() % 2 == 0){
                    median = (sortedValues[mid-1] + sortedValues[mid]) / 2.0f;
                }
                std::vector<float> absDevs;
                absDevs.reserve(values.size());
                for(float v : values){
                    absDevs.push_back(std::abs(v - median));
                }
                std::sort(absDevs.begin(), absDevs.end());
                size_t midDev = absDevs.size() / 2;
                float threshold = absDevs[midDev];
                if(absDevs.size() % 2 == 0){
                    threshold = (absDevs[midDev-1] + absDevs[midDev]) / 2.0f;
                }
                return threshold;
            };

            std::vector<Network> selection;
            selection.reserve(individuals.size());

            meanFitness = 0;
            minFitness = individuals[0].fitness;
            bestFit = individuals[0].fitness;
            maxNetworkSize = individuals[0].innerNodes.size();

            std::vector<int> allIndices(individuals.size());
            std::iota(allIndices.begin(), allIndices.end(), 0);

            const size_t nObjectives = individuals[0].lexicaseObjectives.size(); // used if type=="objectives"
            const size_t nSeeds = individuals[0].fitnessValues.size();          // used if type=="seeds"

            std::vector<int> objectiveOrder(nObjectives);
            std::iota(objectiveOrder.begin(), objectiveOrder.end(), 0);

            std::vector<int> seedOrder(nSeeds);
            std::iota(seedOrder.begin(), seedOrder.end(), 0);

            // type=="seedsRank": epsilon is a number of RANK POSITIONS, not a reward margin.
            // The MAD automatic is deliberately not available here -- the MAD of ranks is
            // always about n/4 regardless of the data, so it would degenerate into "keep the
            // upper half on every test case".
            const int rankTolerance = (epsilon < 0.0f) ? 3 : static_cast<int>(std::lround(epsilon));

            // type=="seedsStandardized": the rewards of each seed are z-transformed over the
            // WHOLE population once per generation -- (value - mean) / sd. epsilon is then a
            // number of standard deviations and means the same thing on every seed and in
            // every generation, no matter how the rewards happen to be distributed.
            //
            // Why not the raw MAD of "seeds": the reward is bimodal (landed ~+200 against
            // crashed ~-200), and a spread estimate on such a distribution mostly measures
            // the distance between the two modes, not the resolution among the near-equal
            // candidates that selection actually has to separate -- epsilon comes out far too
            // large and the test case barely filters. Standardising is the established fix
            // (La Cava et al., epsilon-lexicase); unlike a rank transform it keeps the
            // magnitudes, which is exactly what epsilon is meant to judge.
            //
            // Standardising over the population and not over the shrinking candidate pool is
            // deliberate (the "semi-dynamic" variant): a pool-based sd shrinks along the
            // filter chain, so a fixed epsilon would silently get more permissive with every
            // step.
            std::vector<std::vector<float>> standardizedBySeed;
            if(type == "seedsStandardized"){
                standardizedBySeed.assign(nSeeds, std::vector<float>(individuals.size(), 0.0f));
                for(size_t seedIdx = 0; seedIdx < nSeeds; seedIdx++){
                    double sum = 0.0;
                    for(const auto& ind : individuals) sum += ind.fitnessValues[seedIdx];
                    double mean = sum / static_cast<double>(individuals.size());
                    double sqSum = 0.0;
                    for(const auto& ind : individuals){
                        double d = ind.fitnessValues[seedIdx] - mean;
                        sqSum += d * d;
                    }
                    double sd = std::sqrt(sqSum / static_cast<double>(individuals.size()));
                    // Every individual identical on this seed: the test case carries no
                    // information, leave it at 0 so it filters nobody.
                    if(sd < 1e-9){
                        continue;
                    }
                    for(size_t c = 0; c < individuals.size(); c++){
                        standardizedBySeed[seedIdx][c] =
                            static_cast<float>((individuals[c].fitnessValues[seedIdx] - mean) / sd);
                    }
                }
            }
            // epsilon in standard deviations; negative selects a sensible default.
            const float sigmaTolerance = (epsilon < 0.0f) ? 0.5f : epsilon;

            // type=="objectivesPerSeed": the test cases are the (seed, objective) pairs,
            // flattened as t = seedIdx*nObjPerSeed + objIdx.
            const size_t nObjPerSeed = individuals[0].objectivesPerSeed.empty()
                ? 0 : individuals[0].objectivesPerSeed[0].size();
            std::vector<int> gridOrder;
            if(type == "objectivesPerSeed"){
                if(individuals[0].objectivesPerSeed.empty()){
                    throw std::invalid_argument(
                        "lexicaseSelection(type=\"objectivesPerSeed\"): Network::objectivesPerSeed is "
                        "empty -- call Population::gymnasiumMultiSeed() first");
                }
                gridOrder.resize(individuals[0].objectivesPerSeed.size() * nObjPerSeed);
                std::iota(gridOrder.begin(), gridOrder.end(), 0);
            }

            // One filtering step on one test case: keep everyone within the epsilon/MAD
            // tolerance of the best value. The tolerance is computed per test case, so the
            // very different scales of the five objectives need no normalisation.
            auto filterOnTestCase = [&](std::vector<int>& candidates, auto&& valueOf){
                float best = std::numeric_limits<float>::lowest();
                std::vector<float> values;
                values.reserve(candidates.size());
                for(int c : candidates){
                    float v = valueOf(c);
                    values.push_back(v);
                    if(v > best) best = v;
                }
                float threshold = computeThreshold(values);
                std::vector<int> filtered;
                filtered.reserve(candidates.size());
                for(int c : candidates){
                    if(valueOf(c) >= best - threshold) filtered.push_back(c);
                }
                candidates = std::move(filtered);
            };

            for(int i=0; i<static_cast<int>(individuals.size())-E; i++){
                if(individuals[i].innerNodes.size() > maxNetworkSize){
                    maxNetworkSize = individuals[i].innerNodes.size();
                }

                std::vector<int> candidates = allIndices;

                if(type == "objectives"){
                    // Draw a fresh, independent objective evaluation order for this selection event
                    std::shuffle(objectiveOrder.begin(), objectiveOrder.end(), *generator);

                    for(int objIdx : objectiveOrder){
                        if(candidates.size() <= 1) break;
                        filterOnTestCase(candidates, [&](int c){
                            return individuals[c].lexicaseObjectives[objIdx];
                        });
                    }
                } else if(type == "objectivesPerSeed"){
                    // Fresh, independent order over the whole (seed x objective) grid --
                    // seeds and objectives are shuffled together, not nested, so no seed and
                    // no objective is systematically decided before the others.
                    std::shuffle(gridOrder.begin(), gridOrder.end(), *generator);

                    for(int t : gridOrder){
                        if(candidates.size() <= 1) break;
                        const size_t seedIdx = static_cast<size_t>(t) / nObjPerSeed;
                        const size_t objIdx  = static_cast<size_t>(t) % nObjPerSeed;
                        filterOnTestCase(candidates, [&](int c){
                            return individuals[c].objectivesPerSeed[seedIdx][objIdx];
                        });
                    }
                } else if(type == "seedsStandardized"){
                    // Wie "seeds", aber auf den z-transformierten Werten und mit einer festen
                    // Toleranz in Standardabweichungen statt der MAD-Automatik.
                    std::shuffle(seedOrder.begin(), seedOrder.end(), *generator);

                    for(int seedIdx : seedOrder){
                        if(candidates.size() <= 1) break;
                        const std::vector<float>& column = standardizedBySeed[seedIdx];
                        float best = std::numeric_limits<float>::lowest();
                        for(int c : candidates){
                            if(column[c] > best) best = column[c];
                        }
                        std::vector<int> filtered;
                        filtered.reserve(candidates.size());
                        for(int c : candidates){
                            if(column[c] >= best - sigmaTolerance) filtered.push_back(c);
                        }
                        candidates = std::move(filtered);
                    }
                } else if(type == "seedsRank"){
                    // ─── Rangbasiert je Seed ──────────────────────────────────────────
                    // Wie "seeds", aber der Filter schneidet nach RANG statt nach Rewardwert.
                    //
                    // Warum: Die MAD-Toleranz von "seeds" wird aus den Werten berechnet, und
                    // der Reward ist zweigipflig (gelandet ~+200 gegen abgestuerzt ~-200).
                    // Die MAD misst dort im Wesentlichen den Abstand der beiden Gipfel, nicht
                    // die Aufloesung zwischen den fast gleichen Kandidaten, um die es bei der
                    // Selektion geht -- das Epsilon faellt zu gross aus und der Testfall
                    // trennt kaum. Auf Raengen ist die Schaerfe von der Form und der Skala
                    // des Rewards unabhaengig, und die Seed-Schwierigkeit faellt vollstaendig
                    // heraus: ein schwerer und ein leichter Seed liefern dieselbe
                    // Rangverteilung.
                    //
                    // Die Raenge werden bei JEDEM Filterschritt neu ueber den verbliebenen
                    // Kandidatenpool gebildet. Sonst liessen die spaeteren Testfaelle nach --
                    // wer uebrig ist, liegt ohnehin eng beieinander, und eine feste
                    // Rangtoleranz wuerde alle durchlassen. So bleibt die Pressung ueber die
                    // ganze Kette konstant.
                    std::shuffle(seedOrder.begin(), seedOrder.end(), *generator);

                    for(int seedIdx : seedOrder){
                        if(candidates.size() <= 1) break;

                        // Rangschnitt: die verbliebenen Werte absteigend sortieren und bei
                        // Rang (rankTolerance + 1) abschneiden. Gleichstaende ueberleben
                        // gemeinsam, die Zahl der Ueberlebenden kann also groesser sein.
                        std::vector<float> values;
                        values.reserve(candidates.size());
                        for(int c : candidates){
                            values.push_back(individuals[c].fitnessValues[seedIdx]);
                        }
                        std::sort(values.begin(), values.end(), std::greater<float>());
                        size_t cutIndex = std::min(static_cast<size_t>(std::max(0, rankTolerance)),
                                                   values.size() - 1);
                        float cutoff = values[cutIndex];

                        std::vector<int> filtered;
                        filtered.reserve(candidates.size());
                        for(int c : candidates){
                            if(individuals[c].fitnessValues[seedIdx] >= cutoff){
                                filtered.push_back(c);
                            }
                        }
                        candidates = std::move(filtered);
                    }
                } else { // type == "seeds"
                    // Draw a fresh, independent seed evaluation order for this selection event
                    std::shuffle(seedOrder.begin(), seedOrder.end(), *generator);

                    for(int seedIdx : seedOrder){
                        if(candidates.size() <= 1) break;
                        // Single criterion per seed: maximise the raw reward.
                        filterOnTestCase(candidates, [&](int c){
                            return individuals[c].fitnessValues[seedIdx];
                        });
                    }
                }

                // Pick winner: single survivor, or uniform random tie-break among remaining candidates
                int winnerIdx;
                if(candidates.empty()){
                    // safety fallback: should not happen with non-empty test-case data, but avoids a crash
                    winnerIdx = allIndices[0];
                } else if(candidates.size() == 1){
                    winnerIdx = candidates[0];
                } else {
                    std::uniform_int_distribution<int> tieBreak(0, static_cast<int>(candidates.size())-1);
                    winnerIdx = candidates[tieBreak(*generator)];
                }

                selection.push_back(individuals[winnerIdx]);

                float winnerFitness = individuals[winnerIdx].fitness;
                meanFitness += winnerFitness;
                if(winnerFitness < minFitness) minFitness = winnerFitness;
                if(winnerFitness > bestFit) bestFit = winnerFitness;
            }

            setElite(E, individuals, selection);
            individuals = std::move(selection);
            // set frozenExperience flag back for non-elite
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
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
         * @param useLineageFitness Wenn true, wird zur Auswahl/Rangierung der Elite
         *        `lineageMean` statt der rohen `fitness` verwendet -- aber nur fuer
         *        Individuen, deren `lineageN >= minLineageN` (sonst Fallback auf rohe
         *        Fitness fuer dieses Individuum). Default false = bisheriges Verhalten
         *        (reine rohe Fitness), damit andere Aufrufer (z.B. lexicaseSelection())
         *        unveraendert bleiben.
         * @param minLineageN Mindestanzahl an Lineage-Beobachtungen, ab der lineageMean
         *        statt roher Fitness fuer die Elite-Rangierung genutzt wird (nur relevant,
         *        wenn useLineageFitness=true).
         * @param lineageZ Konfidenz-Multiplikator fuer die untere Konfidenzgrenze (LCB)
         *        des Lineage-Mittelwerts (siehe Network::lineageLCB()). 0.0 (Default) =
         *        reiner lineageMean ohne Unsicherheitsabschlag (bisheriges Verhalten).
         *        Groesseres lineageZ bestraft Linien mit kleinem lineageN/hoher Varianz
         *        staerker -- verhindert, dass frisch ueber minLineageN gekommene
         *        "Gluecks-Neulinge" durch Stichprobenrauschen etablierte Champions
         *        verdraengen (Winner's-Curse-Fix).
         * 
         * @note Elite indices are used to protect elite from mutation operations
         * @note bestFit wird immer aus der rohen Fitness des gewaehlten Elite-Kandidaten
         *       aktualisiert, unabhaengig vom Rangierungskriterium -- Reporting bleibt so
         *       konsistent auf roher Fitness (siehe tournamentSelection()).
         */
        void setElite(int E, const std::vector<Network>& individuals, std::vector<Network>& selection,
                      bool useLineageFitness = false, int minLineageN = 10, float lineageZ = 0.0f){
            indicesElite.clear();

            auto selectionCriterion = [&](const Network& ind) -> float {
                if(useLineageFitness && ind.lineageN >= minLineageN){
                    return ind.lineageLCB(lineageZ);
                }
                return ind.fitness;
            };
            
            std::vector<unsigned int> candidateIndices(individuals.size());
            std::iota(candidateIndices.begin(), candidateIndices.end(), 0);

            for(int counter = 0; counter < E; ++counter){
                float eliteCriterion = std::numeric_limits<float>::lowest();
                unsigned int bestCandIdx = 0;
                for(unsigned int c = 0; c < candidateIndices.size(); ++c){
                    unsigned int idx = candidateIndices[c];
                    float criterion = selectionCriterion(individuals[idx]);
                    if(criterion > eliteCriterion){
                        eliteCriterion = criterion;
                        bestCandIdx = c;
                    }
                }
                unsigned int eliteIndex = candidateIndices[bestCandIdx];
                candidateIndices.erase(candidateIndices.begin() + bestCandIdx); // erase on small int-vector, not Network-vector

                indicesElite.push_back(selection.size()); // because of push_back of elite the index is the old size
                selection.push_back(individuals[eliteIndex]);
                float eliteFit = individuals[eliteIndex].fitness; // rohe Fitness fuer bestFit-Reporting
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
        /**
         * @brief Deckelt die Lineage-Beobachtungszahl (lineageN/lineageSuccessN) NUR bei
         *        Individuen, die diese Generation TATSAECHLICH mutiert oder per Crossover
         *        veraendert wurden (individuals[i].structureChangedThisGen == true) -- das
         *        Mutations-Pendant zum priorCap in crossover(). Soll einmal pro Generation
         *        aufgerufen werden, NACHDEM alle Mutationsoperatoren dieser Generation
         *        gelaufen sind (unabhaengig davon, wie viele/welche Mutationsoperatoren
         *        genutzt wurden -- die Operation ist rein monoton und daher robust gegen
         *        Mehrfachaufruf).
         *
         * @details Ohne diesen (auf tatsaechliche Veraenderung bedingten) Cap wuerde entweder
         * (a) eine reine Mutations-Linie ihre `lineageMean` unbegrenzt akkumulieren und
         * dadurch nach vielen Generationen sehr traege gegenueber neuen Beobachtungen werden,
         * ODER (b) -- falls der Cap blanket auf ALLE nicht-elitaeren Individuen angewendet
         * wuerde, unabhaengig von echter struktureller Veraenderung (frueherer Bug) -- auch
         * unveraenderte Individuen jede Generation unnoetig Historie verlieren, was
         * lineageMean staerker vom aktuellen (verrauschten) Seed-Batch dominieren laesst als
         * beabsichtigt. Der Cap wird daher NUR bei tatsaechlich veraenderten Linien
         * angewendet; unveraenderte Linien behalten ihre volle akkumulierte Historie.
         *
         * Das Flag `structureChangedThisGen` wird nach dem Auslesen (unabhaengig davon, ob
         * gecappt wurde) hier fuer ALLE Individuen zurueckgesetzt, damit die naechste
         * Generation wieder bei false startet.
         *
         * @param priorCap Obergrenze fuer lineageN/lineageSuccessN. Default 5 (wie crossover()).
         */
        void capLineageAfterMutation(int priorCap = 5) {
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                if (!isElite && individuals[i].structureChangedThisGen) {
                    individuals[i].capLineageStats(priorCap);
                }
                individuals[i].structureChangedThisGen = false; // Reset fuer die naechste Generation
            }
        }

        /**
         * @brief Seed-robust default success criterion for updateAdaptiveK().
         *
         * @details
         * Instead of comparing an individual's raw fitness of THIS generation against a
         * single (equally noisy, differently-seeded) prior generation's raw fitness --
         * which would just re-inject the seed-sampling noise this whole mechanism is meant
         * to filter out -- this criterion compares against the individual's own lineage's
         * long-run, uncertainty-aware baseline (Network::lineageLCB()), aggregated over
         * ALL seed-generations this lineage has ever been evaluated on.
         *
         * "Success" = fitness this generation is at or above that robust historical baseline,
         * i.e. the lineage is (still) performing consistently with -- or better than -- its
         * own track record, rather than reacting to a single lucky/unlucky seed draw.
         *
         * @param ind Individual to evaluate.
         * @param z   Confidence multiplier passed to Network::lineageLCB() (larger z = more
         *            conservative baseline). Default 1.0.
         * @return true if lineageN <= 1 (not enough history yet -- neutral/optimistic default),
         *         or if ind.fitness >= ind.lineageLCB(z).
         * @see Network::lineageLCB()
         */
        static bool seedRobustLineageSuccess(const Network& ind, float z = 1.0f) {
            if (ind.lineageN <= 1) return true; // not enough history yet -> neutral, do not count as failure
            return ind.fitness >= ind.lineageLCB(z);
        }

        /**
         * @brief Seed-consistent success criterion for updateAdaptiveK() (bundled DEFAULT).
         *
         * @details
         * Stronger than seedRobustLineageSuccess(): instead of judging success on the
         * AVERAGE fitness across this generation's seed batch (which a single very good or
         * very bad seed can dominate), this criterion requires the improvement to be
         * CONSISTENT across the individual seeds of the current evaluation --
         * i.e. at least `requiredFraction` of this generation's per-seed rewards
         * (Network::fitnessValues) must individually reach or exceed the lineage's robust
         * long-run baseline (Network::lineageLCB()).
         *
         * This directly targets "sampling luck": a mutation that only looks good because it
         * happened to draw a few easy seeds this generation (while still failing hard seeds)
         * is NOT counted as a success, even if the mean fitness looks fine.
         *
         * @param ind               Individual to evaluate.
         * @param requiredFraction  Minimum fraction of per-seed rewards that must be
         *                          >= lineageLCB(z) for the generation to count as a success.
         *                          1.0 = ALL seeds must be consistent, lower values (e.g. 0.8)
         *                          relax this to "most" seeds. Default 1.0.
         * @param z                 Confidence multiplier passed to Network::lineageLCB().
         *                          Default 1.0.
         * @return true if lineageN <= 1 or fitnessValues is empty (not enough history/data yet
         *         -- neutral/optimistic default), otherwise true iff the fraction of per-seed
         *         rewards >= lineageLCB(z) is >= requiredFraction.
         * @see Network::lineageLCB()
         * @see seedRobustLineageSuccess()
         */
        static bool seedConsistentSuccess(const Network& ind, float requiredFraction = 1.0f, float z = 1.0f) {
            if (ind.lineageN <= 1 || ind.fitnessValues.empty()) return true; // not enough history/data yet -> neutral

            float baseline = ind.lineageLCB(z);
            int nAtOrAbove = 0;
            for (float v : ind.fitnessValues) {
                if (v >= baseline) nAtOrAbove++;
            }
            float fraction = static_cast<float>(nAtOrAbove) / static_cast<float>(ind.fitnessValues.size());
            return fraction >= requiredFraction;
        }

        /**
         * @brief Batch-only, absolute-threshold success criterion for updateAdaptiveK().
         *
         * @details
         * Unlike seedConsistentSuccess()/seedRobustLineageSuccess(), this criterion does NOT
         * depend on Network::lineage* stats at all -- it only looks at THIS generation's
         * per-seed rewards (Network::fitnessValues). Useful when lineage tracking is disabled
         * (Population::gymnasiumMultiSeed(..., useLineageFitness=false, ...), e.g. because
         * seed identity is not stable across generations (see uniformDirectionCurriculum)),
         * which would otherwise leave lineageN permanently at 0 and make seedConsistentSuccess()
         * always return its neutral "true" default -- silently saturating the success rate at
         * ~100% and driving adaptiveK_ to kMin/kMax immediately.
         *
         * Success requires at least `requiredFraction` of this generation's per-seed rewards
         * to individually reach or exceed a fixed, absolute `threshold` -- i.e. "(almost) all
         * seeds this generation were (near-)perfect", with no dependency on any running history.
         *
         * @param ind               Individual to evaluate.
         * @param threshold         Absolute per-seed reward threshold to count as "successful"
         *                          for that seed. Default 900.0 (near-perfect for the 1000-cap
         *                          LunarLander fitness scale used in this codebase).
         * @param requiredFraction  Minimum fraction of per-seed rewards that must be >= threshold
         *                          for the generation to count as a success. 1.0 = ALL seeds.
         *                          Default 1.0.
         * @return true if fitnessValues is empty (no data yet -- neutral/optimistic default),
         *         otherwise true iff the fraction of per-seed rewards >= threshold is
         *         >= requiredFraction.
         * @see seedConsistentSuccess()
         */
        static bool batchThresholdSuccess(const Network& ind, float threshold = 900.0f, float requiredFraction = 1.0f) {
            if (ind.fitnessValues.empty()) return true; // no data yet -> neutral

            int nAtOrAbove = 0;
            for (float v : ind.fitnessValues) {
                if (v >= threshold) nAtOrAbove++;
            }
            float fraction = static_cast<float>(nAtOrAbove) / static_cast<float>(ind.fitnessValues.size());
            return fraction >= requiredFraction;
        }

        /**
         * @brief Continuous, scale-independent seed-consistency score for updateAdaptiveK() (bundled DEFAULT).
         *
         * @details
         * Unlike seedConsistentSuccess()/batchThresholdSuccess(), which are BINARY and (for
         * batchThresholdSuccess()) depend on an absolute fitness threshold that is only
         * reachable late in training (early/mid-training generations would then always count
         * as "failure" regardless of how seed-consistent the individual actually already is),
         * this returns a CONTINUOUS score in [0,1] purely from the RELATIVE spread of THIS
         * generation's per-seed rewards (Network::fitnessValues) -- meaningful from generation 1
         * onward, at any absolute fitness level:
         *
         *     consistency = max(0, min(fitnessValues)) / max(fitnessValues)
         *
         * 1.0 = the worst seed this generation did just as well as the best seed (perfectly
         * consistent, regardless of whether that level is low or high). Values near 0 mean a
         * severe outlier seed relative to the best one (fragile/non-robust structure).
         *
         * Averaged over the non-elite population by updateAdaptiveK() and then smoothed over
         * `windowSize` generations via its rolling window -- i.e. "consistency over multiple
         * generations", not just a single noisy snapshot.
         *
         * @param ind Individual to evaluate.
         * @return 1.0 if fewer than 2 fitnessValues or max(fitnessValues) <= 0 (not enough
         *         data / degenerate case -- neutral/optimistic default), otherwise
         *         max(0, min(fitnessValues)) / max(fitnessValues), clamped to [0,1].
         * @see batchThresholdSuccess()
         * @see seedConsistentSuccess()
         */
        static float seedConsistencyRatio(const Network& ind) {
            if (ind.fitnessValues.size() < 2) return 1.0f; // not enough seeds this gen -> neutral

            float minV = *std::min_element(ind.fitnessValues.begin(), ind.fitnessValues.end());
            float maxV = *std::max_element(ind.fitnessValues.begin(), ind.fitnessValues.end());
            if (maxV <= 0.0f) return 1.0f; // degenerate (all non-positive) -> neutral, avoid div-by-zero/negative ratio

            minV = std::max(minV, 0.0f); // clamp negative worst-seed rewards to 0 so the ratio stays in [0,1]
            return std::min(1.0f, minV / maxV);
        }



        /**
         * @brief Self-adaptive mutation-strength k, following Rechenberg's 1/5-success-rule,
         *        with a fully exchangeable, seed-robust success criterion.
         *
         * @details
         * Must be called once per generation, AFTER this generation's evaluation/selection
         * (gymnasiumMultiSeed()/lexicaseSelection() etc. -- so that individuals[i].fitness
         * reflects the CURRENT generation) and BEFORE this generation's mutation operators
         * are applied (callEdgeMutation()/callBoundaryMutation*() should then be called with
         * `k = pop.adaptiveK_`).
         *
         * **Procedure**:
         * 1. For every non-elite individual, evaluates `successCriterion(individual)` -- a
         *    score in [0,1] (bool predicates are implicitly treated as 1.0/0.0).
         * 2. Records the population-average score this generation in a rolling window of
         *    size `windowSize` (successRateHistory_) -- this is what provides "consistency
         *    over multiple generations" rather than reacting to a single noisy generation.
         * 3. If `useRechenberg == true`: compares the window-averaged score against
         *    `targetSuccessRate` and multiplies adaptiveK_ by `incFactor` or `decFactor`,
         *    then clamps to [kMin, kMax]. Which factor is applied when depends on `invert`:
         *    - `invert == false` (classical Rechenberg, convergence-speed-oriented):
         *      score > target -> mutation too weak -> `incFactor` (grow k);
         *      score < target -> mutation too disruptive -> `decFactor` (shrink k).
         *    - `invert == true` (stability-oriented, recommended when the success criterion
         *      is already seed-robust, e.g. seedConsistencyRatio()): score > target
         *      -> already consistently succeeding -> `decFactor` (shrink k, preserve/stabilize
         *      the working structure); score < target -> not consistently succeeding
         *      -> `incFactor` (grow k, explore out of the insufficient structure).
         * 4. If `useRechenberg == false`: history is still recorded (for diagnostics/plots),
         *    but adaptiveK_ is left untouched -- use setAdaptiveK() to set a fixed value
         *    manually in that case.
         *
         * The scoring function is fully exchangeable: pass any `(const Network&) -> float`
         * (or legacy `-> bool`) callable. The bundled default (seedConsistencyRatio()) is a
         * CONTINUOUS, scale-independent measure of this generation's per-seed spread
         * (worst/best seed ratio) -- meaningful from generation 1 onward, unlike the
         * threshold-based alternatives (batchThresholdSuccess()) which require near-maximal
         * absolute fitness to ever register as "successful" and therefore stay stuck at one
         * extreme for as long as that absolute level hasn't been reached yet. The
         * lineage-based alternatives (seedConsistentSuccess()/seedRobustLineageSuccess())
         * remain available but require Population::gymnasiumMultiSeed(..., useLineageFitness=true, ...).
         *
         * @param successCriterion  Per-individual scoring function, [0,1] (or bool). Default: seedConsistencyRatio().
         * @param useRechenberg     If false, only records history; adaptiveK_ stays unchanged. Default true.
         * @param invert            If true, swaps which factor is applied above/below target
         *                          (stability-first: success -> shrink k, struggling -> grow k).
         *                          Default false (classical Rechenberg direction).
         * @param windowSize        Number of past generations kept in the rolling score window
         *                          -- the "consistency over multiple generations" smoothing. Default 10.
         * @param targetSuccessRate Target (window-averaged) score. Default 0.2 (= classical 1/5 Rechenberg optimum).
         * @param incFactor         Multiplicative "grow k" factor, applied above target when
         *                          `invert==false`, or below target when `invert==true`. Default 1.22.
         * @param decFactor         Multiplicative "shrink k" factor, applied below target when
         *                          `invert==false`, or above target when `invert==true`. Default 0.82.
         * @param kMin              Lower clamp for adaptiveK_. Default 0.1.
         * @param kMax              Upper clamp for adaptiveK_. Default 5.0.
         *
         * @see seedConsistencyRatio()
         * @see batchThresholdSuccess()
         * @see seedConsistentSuccess()
         * @see adaptiveK_
         */
        void updateAdaptiveK(
                SuccessCriterion successCriterion = [](const Network& ind){ return seedConsistencyRatio(ind); },
                bool useRechenberg = true,
                bool invert = false,
                int windowSize = 10,
                float targetSuccessRate = 0.2f,
                float incFactor = 1.22f,
                float decFactor = 0.82f,
                float kMin = 0.1f,
                float kMax = 5.0f)
        {
            float sumScore = 0.0f;
            int total = 0;
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end())
                    continue; // elite was not mutated -> excluded from the success statistic
                sumScore += successCriterion(individuals[i]);
                total++;
            }
            if (total == 0) return; // nothing to adapt on (e.g. everyone is elite)

            float rateThisGen = sumScore / static_cast<float>(total);
            successRateHistory_.push_back(rateThisGen);
            while (static_cast<int>(successRateHistory_.size()) > windowSize) {
                successRateHistory_.pop_front();
            }

            if (!useRechenberg) return; // diagnostics only, adaptiveK_ stays as manually set

            float meanRate = 0.0f;
            for (float r : successRateHistory_) meanRate += r;
            meanRate /= static_cast<float>(successRateHistory_.size());

            bool aboveTarget = meanRate > targetSuccessRate;
            bool belowTarget = meanRate < targetSuccessRate;
            if (invert) std::swap(aboveTarget, belowTarget);

            if (aboveTarget) {
                adaptiveK_ *= incFactor;
            } else if (belowTarget) {
                adaptiveK_ *= decFactor;
            }
            adaptiveK_ = std::min(kMax, std::max(kMin, adaptiveK_));
        }


        void callEdgeMutation(float probInnerNodes, float probStartNode, bool justUsedNodes = false, float k = 0.0f){
            for(int i=0; i<individuals.size(); i++){

                int N;
                if(k > 0.0f){
                    N = individuals[i].countEdges(justUsedNodes);
                }
                else {
                    N = 0;
                }

                if(std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()){// preventing elite
                    bool changed = false;
                    for(auto& node : individuals[i].innerNodes){

                        if (node.frozen > 0) continue;

                        if(justUsedNodes == true && node.used == false){
                            continue;
                        } else {
                            if(node.edgeMutation(probInnerNodes, individuals[i].innerNodes.size(), k, N, &individuals[i].innerNodes)) changed = true;
                        }
                    }
                    if(individuals[i].startNode.edgeMutation(probStartNode, individuals[i].innerNodes.size(), k, N, &individuals[i].innerNodes)) changed = true;
                    if(changed) individuals[i].structureChangedThisGen = true;
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
        void applyBoundaryMutation(FuncMutation&& func, bool justUsedNodes = false, float k = 0.0f, int N = 1) {
            for (int i = 0; i < individuals.size(); ++i) {

                if (k > 0.0f) {
                    N = individuals[i].countEdges(justUsedNodes);
                } else {
                    N = 0;
                }

                if (std::find(indicesElite.begin(), indicesElite.end(), i) == indicesElite.end()) {
                    additionalMutationParam amp;
                    amp.networkSize = individuals[i].innerNodes.size();
                    bool changed = false;
                    for (auto& node : individuals[i].innerNodes) {
                        
                        if (node.frozen > 0) continue;

                        if (node.type == "J" || node.type == "JE") {
                           if (justUsedNodes == true) {
                                if (node.used == true) {
                                    if (func(node, amp, justUsedNodes, k, N)) changed = true;
                                }
                            } else { 
                                if (func(node, amp, justUsedNodes, k, N)) changed = true;
                            }
                        }
                    }
                    if (changed) individuals[i].structureChangedThisGen = true;
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
        void callBoundaryMutationUniform(const float probability, bool justUsedNodes = false, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes, float k, int N){ 
                return node.boundaryMutationUniform(probability, k, N);
            }, justUsedNodes, k); 
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
        void callBoundaryMutationNormal(const float probability, const float sigma, bool justUsedNodes, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes, float k, int N){
                return node.boundaryMutationNormal(probability, sigma, k, N);
            }, justUsedNodes, k); 
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
        void callBoundaryMutationNetworkSizeDependingSigma(const float probability, const float sigma, bool justUsedNodes, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam& amp, bool justUsedNodes, float k, int N){
                float sigmaNew = sigma * (1/log(amp.networkSize));
                return node.boundaryMutationNormal(probability, sigmaNew, k, N);
            }, justUsedNodes, k);
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
        void callBoundaryMutationEdgeSizeDependingSigma(const float probability, const float sigma, bool justUsedNodes, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes, float k, int N){
                float sigmaNew = sigma * (1/log(node.edges.size()));
                return node.boundaryMutationNormal(probability, sigmaNew, k, N);
            }, justUsedNodes, k);
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
        void callBoundaryMutationFractal(const float probability, std::vector<float> minF, std::vector<float> maxF, bool justUsedNodes, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes, float k, int N){
                return node.boundaryMutationFractal(probability, minF, maxF);
            }, justUsedNodes, k);
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
         * @param propability Probability (in [0.0, 1.0]), interpreted per crossover type. Default is 1
         *  (i.e. crossover is always applied). Values outside [0.0, 1.0] throw std::invalid_argument,
         *  since std::bernoulli_distribution would otherwise be undefined behaviour.
         *  - "uniform": per NODE POSITION -- each of the min(size1, size2) positions is exchanged with
         *    this probability, so it controls HOW MUCH of the genotype is exchanged per pair.
         *  - "semantic": TWO rates. This value gates the PAIR (one draw decides whether the pair
         *    recombines at all); `nodeExchangeRate` then decides how many of the role-matched
         *    nodes inside that pair are transferred. Passing a negative nodeExchangeRate couples
         *    the two, i.e. the effective rate per matched node becomes propability^2.
         *    recombined at all; if it fails the pair is left completely untouched. What is exchanged
         *    afterwards (cutpoint, subnetwork depth, cluster blocks) stays uniformly random.
         *  - "seedSpecialist" / "seedSpecialistAppend" / "seedSpecialistReplace": per INDIVIDUAL --
         *    one draw per host decides whether that host receives a transplant this generation.
         *    Drawn before the donor search, so a low value also skips the O(populationSize) search
         *    for that host.
         * @param type Type of the crossover:
         *  - "innovation": NEAT-style crossover aligned on Node::innovationID (historical markings,
         *    Stanley & Miikkulainen 2002) instead of the array position. Genes present in BOTH parents
         *    (matching) are inherited from the donor with probability 0.5; genes present in only one
         *    parent (disjoint/excess) are taken from the FITTER parent, measured as the mean over
         *    fitnessValues (falling back to Network::fitness when per-seed data is missing). The fitter
         *    parent survives unchanged and the weaker one is overwritten by the child, so the operator is
         *    one-directional per pair; if one parent is elite, the elite always donates. Edge targets of
         *    inherited nodes are translated donor-position -> innovationID -> recipient-position, i.e. the
         *    LINKAGE between genes is transferred rather than an index pattern; targets whose gene is
         *    absent in the recipient are repaired like in the other types. Requires unique markings per
         *    individual, which ensureInnovationIDs() establishes on the fly.
         *  - "uniform": selects each node and exchanges them with given probability
         *  - "onepoint": draws a random cutpoint from the genotype and exchanges all nodes until this point
         *  - "randomWidth": exchanges subnetworks of different widths where all succesor nodes of a randomly selected node are exchanged
         *    subgraph. A random number of cluster-block pairs (1..min(#clusters1, #clusters2)) is drawn and
         *    exchanged, one cluster picked independently (no size-matching -- asymmetric swaps of many vs. few
         *    nodes are intentional) from each parent per pair. Requires both parents to carry valid, up-to-date
         *    another crossover type). Dangling boundary edges after the swap(s) are repaired via changeFalseEdges(),
         *    exactly as for "randomWidth".
         *  @param traversalNeighbor If true, the crossover is only applied to pairs of individuals that are neighbors in the traversal space 
         *  calculated by traverseCounter.
         *  @param lowerBoundTraversalCounter Lower bound for traverseCounter ratio to consider individuals as neighbors (used if traversalNeighbor is true)
         *  @param upperBoundTraversalCounter Upper bound for traverseCounter ratio to consider individuals as neighbors (used if traversalNeighbor is true)
          *  @param boundaryTolerance Only used by type="semantic": maximum normalised distance
          *         between the inner boundaries of two same-role nodes for them to still count
          *         as the same gene. 1.0 (default) matches everything within a role; smaller
          *         values additionally require that both cut the feature at a similar place.
          *  @param nodeExchangeRate Only used by type="semantic": rate at which a MATCHED node is
          *         transferred, inside a pair that already passed the per-pair gate
          *         (`propability`). Negative (default) means "use propability", reproducing the
          *         historical behaviour where one value was applied twice.
          *  @param matchOnlyUsed Only used by type="semantic": if true, nodes never entered
          *         during the last evaluation (Node::used == false) take no part in the
          *         matching, so dormant material is not exchanged.
         * 
         * @note tournamentSelection() must have been called to set indicesElite
         * @note Only nodes up to min(size1, size2) can be exchanged for "uniform" and "onepoint" due to position-based matching
         *  - "seedSpecialistReplace": same donor search and same donor-side sub-graph as
         *    "seedSpecialistAppend" (see below), but the sub-graph does not become a dormant
         *    appendix -- it OVERWRITES the host's own deficit-exclusive sub-graph, i.e. the nodes
         *    the host traverses ONLY on the deficit seeds and not on the seeds where the host
         *    itself is the better one. The shared backbone is therefore preserved, and with it the
         *    host's strength on its good seeds; only the seed-specific structure that fails is
         *    swapped out. If that set is empty (the host uses the same nodes everywhere), the host
         *    is left unchanged. Because this variant is destructive for the host, ELITE hosts are
         *    skipped (unlike the additive variant). The donor is never modified -- it only hands
         *    out copies, so one donor can serve several hosts in the same generation. Sub-graphs of
         *    unequal size are handled exactly like the elite branch of "randomWidth"
         *    (addOverhangNodes()/deleteOverhangNodes()), so the host grows or shrinks by the size
         *    difference. The replaced nodes are wired into the existing graph and are therefore
         *    active immediately: they carry NO transplantBlockID/isBlockEntry, and the host's
         *    them). Like the other types it obeys crossoverProtection via generationReceived.
         *  - "seedSpecialist" / "seedSpecialistAppend" (identical, the short name is kept for
         *    backward compatibility): for EVERY individual (host, including elite) independently searches the whole
         *    population for the single donor individual maximizing the summed positive per-seed fitness
         *    difference (donor.fitnessValues[s] - host.fitnessValues[s], summed over all s where positive).
         *    If such a donor exists, the union of nodes the donor actually traversed (Network::visitedNodesPerSeed,
         *    see gymnasiumMultiSeed()) across those deficit seeds is copied (never removed from the donor) and
         *    APPENDED as new, additional nodes to the host -- the host's existing nodes/edges are left completely
         *    untouched. The appended nodes are protected from further crossover via generationReceived (same
         *    crossoverProtection mechanism as the other types) and tagged with a shared Node::transplantBlockID;
         *    exactly one of them (the donor's own traversal start node, always part of the copied set) is marked
         *    Node::isBlockEntry. Because the block is appended without being wired into any existing edge, it
         *    stays a dormant, fitness-neutral addition (analogous to "junk DNA" from callAddDelNodes()) until a
         *    LATER mutation call happens to redirect an existing host edge onto the block's entry node --
         *    Node::changeEdge()/edgeMutation() are extended so that only the entry node (never an interior node)
         *    of a block can be chosen as a mutation target from outside the block. Once entered this way, the
         *    block behaves like any other part of the network (its own internal edges mutate normally, are not
         *    frozen). This type has its OWN partner-search/insertion logic and completely bypasses the
         *    shuffle-based pairing loop used by the other types (see @note below).
         */

        void crossover(
                float propability = 1, 
                std::string type = "", 
                int currentGeneration = 0, 
                int crossoverProtection = 0,
                bool traversalNeighbor = false, 
                float lowerBoundTraversalCounter = 0.9, 
                float upperBoundTraversalCounter = 1.1,
                int lineagePriorCap = 5,
                float boundaryTolerance = 1.0f,
                bool matchOnlyUsed = false,
                float nodeExchangeRate = -1.0f
                ){

            // type="semantic" uses TWO rates, and they used to be the same value applied
            // twice: `propability` gated the pair AND decided every node inside it, so the
            // effective exchange rate per matched node was propability^2 (0.05 -> 0.0025).
            // They are separate now. nodeExchangeRate < 0 keeps the old coupling, so every
            // existing call reproduces its previous behaviour bit for bit.
            const float nodeRate = (nodeExchangeRate < 0.0f) ? propability : nodeExchangeRate;
            std::bernoulli_distribution distributionNodeExchange(nodeRate);

            crossoverPairsApplied = 0;
            crossoverNodesMatched = 0;
            crossoverNodesExchanged = 0;
            crossoverNodesSkipped = 0;

            if(propability < 0.0f || propability > 1.0f){
                throw std::invalid_argument("crossover(): propability must be in [0.0, 1.0]");
            }

            std::bernoulli_distribution distributionBernoulli(propability);

            // ─── "seedSpecialist*": own partner search + insertion logic, fully independent ───
            // of the shuffle-based pairing below (per-individual best-donor search over the whole
            // population -- see class-level docstring for details). Both variants share the donor
            // search and the donor-side sub-graph; they differ only in HOW that sub-graph enters
            // the host: "…Append" adds it as a dormant block, "…Replace" overwrites the host's own
            // deficit-exclusive sub-graph with it.
            // Early return keeps every other crossover type's code path byte-for-byte unchanged.
            const bool seedSpecialistAppend  = (type == "seedSpecialist" || type == "seedSpecialistAppend");
            const bool seedSpecialistReplace = (type == "seedSpecialistReplace");

            if(seedSpecialistAppend || seedSpecialistReplace){
                for(int hostIdx = 0; hostIdx < static_cast<int>(individuals.size()); hostIdx++){
                    auto& host = individuals[hostIdx];
                    if(host.fitnessValues.empty()) continue; // no per-seed data yet this generation

                    // Elite ist nur bei der ERSETZENDEN Variante geschuetzt: dort verliert der Host
                    // eigenes Genmaterial und koennte dabei genau die Fitness einbuessen, die der
                    // Elitismus bewahren soll. Das additive Anhaengen ist fitness-neutral (dormanter
                    // Block) und laeuft deshalb weiterhin fuer jedes Individuum inklusive Elite.
                    if(seedSpecialistReplace &&
                       std::find(indicesElite.begin(), indicesElite.end(), hostIdx) != indicesElite.end()){
                        continue;
                    }

                    // Per-INDIVIDUAL application gate: with probability (1 - propability) this host
                    // receives no transplant this generation. Drawn before the (expensive) donor
                    // search over the whole population, so a low propability also saves the search.
                    if(!distributionBernoulli(*generator)) continue;

                    int bestDonorIdx = -1;
                    float bestScore = 0.0f; // must stay > 0 to be accepted (donor must help on >=1 seed)
                    std::vector<int> bestDeficitSeeds;

                    for(int donorIdx = 0; donorIdx < static_cast<int>(individuals.size()); donorIdx++){
                        if(donorIdx == hostIdx) continue;
                        auto& donor = individuals[donorIdx];
                        if(donor.fitnessValues.size() != host.fitnessValues.size()) continue; // seed batches must line up
                        if(donor.visitedNodesPerSeed.size() != donor.fitnessValues.size()) continue; // tracking not (yet) populated

                        float score = 0.0f;
                        std::vector<int> deficitSeeds;
                        for(size_t s = 0; s < host.fitnessValues.size(); s++){
                            float diff = donor.fitnessValues[s] - host.fitnessValues[s];
                            if(diff > 0.0f){
                                score += diff;
                                deficitSeeds.push_back(static_cast<int>(s));
                            }
                        }

                        if(score > bestScore){
                            bestScore = score;
                            bestDonorIdx = donorIdx;
                            bestDeficitSeeds = std::move(deficitSeeds);
                        }
                    }

                    if(bestDonorIdx == -1) continue; // no individual improves on any seed -> nothing to transplant

                    auto& donor = individuals[bestDonorIdx];

                    // Union of nodes the donor actually traversed on its deficit seeds -- the donor's
                    // "active sub-graph" for exactly the seeds where it outperforms the host.
                    std::unordered_set<int> subGraphSet;
                    for(int s : bestDeficitSeeds){
                        for(int n : donor.visitedNodesPerSeed[s]){
                            if(n >= 0 && n < static_cast<int>(donor.innerNodes.size())) subGraphSet.insert(n);
                        }
                    }
                    if(subGraphSet.empty()) continue;

                    std::vector<int> subGraphNodes(subGraphSet.begin(), subGraphSet.end());
                    std::sort(subGraphNodes.begin(), subGraphNodes.end());

                    // ─── Variante "seedSpecialistReplace": ersetzen statt anhaengen ───────────
                    // Der Donor-Sub-Graph (oben, identisch zur additiven Variante) ueberschreibt
                    // den DEFIZIT-EXKLUSIVEN Sub-Graphen des Hosts: die Knoten, die der Host NUR
                    // auf den Defizit-Seeds durchlaeuft, nicht aber auf den Seeds, auf denen er
                    // selbst besser ist. Damit bleibt das gemeinsam genutzte Backbone -- und
                    // damit die Staerke des Hosts auf seinen guten Seeds -- erhalten; ersetzt
                    // wird nur die seed-spezifische Teilstruktur, die dort versagt. Ist diese
                    // Menge leer (der Host benutzt ueberall dieselben Knoten), gibt es nichts
                    // seed-Spezifisches zu ersetzen und der Host bleibt unveraendert.
                    if(seedSpecialistReplace){
                        if(host.visitedNodesPerSeed.size() != host.fitnessValues.size()) continue;

                        std::vector<char> isDeficitSeed(host.fitnessValues.size(), 0);
                        for(int s : bestDeficitSeeds) isDeficitSeed[s] = 1;

                        std::unordered_set<int> hostDeficitNodes, hostOtherNodes;
                        for(size_t s = 0; s < host.visitedNodesPerSeed.size(); s++){
                            auto& target = isDeficitSeed[s] ? hostDeficitNodes : hostOtherNodes;
                            for(int n : host.visitedNodesPerSeed[s]){
                                if(n >= 0 && n < static_cast<int>(host.innerNodes.size())) target.insert(n);
                            }
                        }

                        std::vector<int> hostBlock;
                        for(int n : hostDeficitNodes){
                            if(hostOtherNodes.find(n) == hostOtherNodes.end()) hostBlock.push_back(n);
                        }
                        if(hostBlock.empty()) continue; // keine defizit-exklusive Teilstruktur vorhanden
                        std::sort(hostBlock.begin(), hostBlock.end()); // Voraussetzung von add/deleteOverhangNodes()

                        // Ab hier exakt das Vorgehen des Elite-Zweigs von "randomWidth": einseitige
                        // Uebernahme (der Donor gibt nur Kopien ab und bleibt unveraendert), gleiche
                        // Ueberhang-Behandlung fuer unterschiedlich grosse Sub-Graphen.
                        const size_t minSubNodes = std::min(subGraphNodes.size(), hostBlock.size());
                        std::unordered_map<int, int> replaceMap =
                            initNodeSwapMap(subGraphNodes, hostBlock, static_cast<int>(host.innerNodes.size()));

                        for(size_t j = 0; j < minSubNodes; j++){
                            auto& node = host.innerNodes[hostBlock[j]];
                            node = donor.innerNodes[subGraphNodes[j]];
                            node.generationReceived = currentGeneration;
                            // Aktiv eingebaut statt dormant: keine Block-Kennung, damit
                            // Node::changeEdge() die Knoten nicht als "nur ueber den Entry-Knoten
                            // erreichbar" behandelt (siehe Node::transplantBlockID/isBlockEntry).
                            node.transplantBlockID = -1;
                            node.isBlockEntry = false;
                            node.frozen = 0;
                            // used wird bewusst auf false gesetzt und NICHT vom Donor uebernommen:
                            // der Knoten lag auf einem aktiven Pfad des SPENDERS, im Empfaenger ist
                            // er dagegen noch nie durchlaufen worden. Wuerde er als used markiert
                            // bleiben, waere er fuer die Loeschbranche von addDelNodes() (die nur
                            // used == false entfernt) dauerhaft unsichtbar -- auch dann, wenn der
                            // Host ihn nie erreicht. Genau darueber sammeln sich sonst Generation
                            // fuer Generation nie durchlaufene Transplantat-Knoten an, ohne dass
                            // die junk-Quote sie je abbauen kann.
                            //
                            // Vor verfruehtem Loeschen schuetzt stattdessen generationReceived in
                            // Verbindung mit crossoverProtection: solange currentGeneration -
                            // generationReceived < crossoverProtection ist, ueberspringt
                            // addDelNodes() den Knoten (siehe Network::addDelNodes()). Der Knoten
                            // ueberlebt damit garantiert bis zur naechsten Evaluation, die sein
                            // used-Flag anhand der tatsaechlichen Traversierung im Host setzt.
                            // ACHTUNG: Mit crossoverProtection == 0 entfaellt dieser Schutz und das
                            // direkt nachfolgende callAddDelNodes() koennte das Transplantat noch in
                            // derselben Generation wieder entfernen.
                            node.used = false;
                            // traverseCounter/lastVisitStep gehoeren dem Donor und wuerden die
                            // Statistik des Hosts verfaelschen.
                            node.traverseCounter = 0;
                            node.lastVisitStep = 0;
                        }

                        std::vector<int> replaceIndices;
                        replaceIndices.reserve(replaceMap.size());
                        for(auto const& [key, val] : replaceMap) replaceIndices.push_back(val);

                        if(subGraphNodes.size() > hostBlock.size()){
                            const size_t sizeBeforeOverhang = host.innerNodes.size();
                            addOverhangNodes(subGraphNodes, hostBlock, donor, host, true, currentGeneration);
                            for(size_t k = sizeBeforeOverhang; k < host.innerNodes.size(); k++){
                                host.innerNodes[k].transplantBlockID = -1;
                                host.innerNodes[k].isBlockEntry = false;
                                host.innerNodes[k].frozen = 0;
                                // Gleiche Begruendung wie beim Ersetzen oben: addOverhangNodes()
                                // kopiert die Donor-Knoten samt used-Flag. Die Ueberhang-Knoten sind
                                // im Host aber ebenso wenig durchlaufen worden wie die ersetzten --
                                // und da genau dieser Zweig den Host wachsen laesst, waere ein
                                // uebernommenes used == true hier die Hauptquelle dauerhaft
                                // unloeschbarer Knoten. generationReceived (von addOverhangNodes()
                                // gesetzt) schuetzt sie bis zur naechsten Evaluation.
                                host.innerNodes[k].used = false;
                                host.innerNodes[k].traverseCounter = 0;
                                host.innerNodes[k].lastVisitStep = 0;
                            }
                        }

                        host.remapNodeIdsAndEdges(replaceMap, replaceIndices, false);

                        if(subGraphNodes.size() < hostBlock.size()){
                            deleteOverhangNodes(hostBlock, subGraphNodes, host);
                        }

                        host.changeFalseEdges();

                        // Positionen im Host haben sich verschoben (Ueberhang angehaengt bzw.
                        // Sie werden verworfen statt mitgefuehrt: callFindTransitionClusters()
                        // berechnet sie ohnehin jede Generation neu, und ein verschobener Vektor
                        // schuetzt in addDelNodes() die falschen Knoten vor dem Loeschen.

                        host.nCrossovers += 1;
                        host.structureChangedThisGen = true;
                        host.blendLineageWith(donor, lineagePriorCap); // nur der Host uebernimmt (Donor unveraendert)
                        continue;
                    }

                    // Build old(donor index) -> new(host index) map for the appended copies.
                    std::unordered_map<int, int> swapMap;
                    swapMap.reserve(subGraphNodes.size());
                    size_t baseIndex = host.innerNodes.size();
                    for(size_t k = 0; k < subGraphNodes.size(); k++){
                        swapMap[subGraphNodes[k]] = static_cast<int>(baseIndex + k);
                    }

                    // Append deep copies of the donor's sub-graph nodes -- the donor itself is never modified.
                    std::vector<int> newIndices;
                    newIndices.reserve(subGraphNodes.size());
                    for(int oldIdx : subGraphNodes){
                        host.innerNodes.push_back(donor.innerNodes[oldIdx]);
                        newIndices.push_back(static_cast<int>(host.innerNodes.size()) - 1);
                    }

                    int blockID = nextTransplantBlockID++;
                    for(int ni : newIndices){
                        auto& node = host.innerNodes[ni];
                        node.id = ni;
                        node.generationReceived = currentGeneration; // protected via crossoverProtection, like the other types
                        node.transplantBlockID = blockID;
                        node.isBlockEntry = false;
                        node.used = false;         // dormant: not (yet) part of any traversal in the host
                        node.traverseCounter = 0;
                        node.lastVisitStep = 0;
                        node.frozen = 0;            // internal edges stay normally mutable once integrated
                    }

                    // Remap internal edges to the new host indices; any edge leaving the copied sub-graph
                    // (e.g. an untraveled branch of a judgment node) is redirected to a random valid host
                    // node, same repair strategy as changeFalseEdges() for the other crossover types.
                    for(int ni : newIndices){
                        auto& node = host.innerNodes[ni];
                        for(auto& e : node.edges){
                            auto it = swapMap.find(e);
                            if(it != swapMap.end()){
                                e = it->second;
                            } else {
                                e = node.changeEdge(static_cast<int>(host.innerNodes.size()), e, &host.innerNodes);
                            }
                        }
                    }

                    // The donor's own traversal always starts at the same node (its startNode.edges[0]),
                    // so it is guaranteed to be part of every seed's visited set -- this is the single
                    // node through which the block may later be entered via mutation.
                    auto entryIt = swapMap.find(donor.startNode.edges[0]);
                    if(entryIt != swapMap.end()){
                        host.innerNodes[entryIt->second].isBlockEntry = true;
                    } else if(!newIndices.empty()){
                        host.innerNodes[newIndices.front()].isBlockEntry = true; // defensive fallback, should not trigger
                    }

                    host.nCrossovers += 1;
                    host.structureChangedThisGen = true;
                    host.blendLineageWith(donor, lineagePriorCap); // only host absorbs donor's lineage stats (donor unaffected)
                }
                return;
            }

            int nNodesToExchange;
            std::vector<unsigned int> inds;
            for(int i=0; i<individuals.size(); i++){
                inds.push_back(i);
            }
            std::shuffle(inds.begin(), inds.end(), *generator);

            // Vermischt die Lineage-Statistik zweier Individuen SYMMETRISCH (beide erhalten
            // Gene voneinander), ohne dass die Reihenfolge zweier nacheinander ausgefuehrter
            // blendLineageWith()-Aufrufe zu einer Asymmetrie fuehrt (sonst wuerde der zweite
            // Aufruf bereits die veraenderten Werte des ersten Individuums sehen). Dazu werden
            // zunaechst beide Original-Werte zwischengespeichert und danach beide Individuen
            // unabhaengig voneinander aktualisiert.
            auto blendLineageBidirectional = [lineagePriorCap](Network& a, Network& b){
                float aMean = a.lineageMean, bMean = b.lineageMean;
                int aN = a.lineageN, bN = b.lineageN;
                float aSuccessRate = a.lineageSuccessRate, bSuccessRate = b.lineageSuccessRate;
                int aSuccessN = a.lineageSuccessN, bSuccessN = b.lineageSuccessN;

                int aNCapped = std::min(aN, lineagePriorCap), bNCapped = std::min(bN, lineagePriorCap);
                int totalN = std::max(1, aNCapped + bNCapped);
                float pooledMean = (aNCapped * aMean + bNCapped * bMean) / static_cast<float>(totalN);

                int aSuccessNCapped = std::min(aSuccessN, lineagePriorCap), bSuccessNCapped = std::min(bSuccessN, lineagePriorCap);
                int totalSuccessN = std::max(1, aSuccessNCapped + bSuccessNCapped);
                float pooledSuccessRate = (aSuccessNCapped * aSuccessRate + bSuccessNCapped * bSuccessRate) / static_cast<float>(totalSuccessN);

                a.lineageMean = pooledMean; a.lineageN = aNCapped + bNCapped;
                a.lineageSuccessRate = pooledSuccessRate; a.lineageSuccessN = aSuccessNCapped + bSuccessNCapped;
                b.lineageMean = pooledMean; b.lineageN = aNCapped + bNCapped;
                b.lineageSuccessRate = pooledSuccessRate; b.lineageSuccessN = aSuccessNCapped + bSuccessNCapped;
            };

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

                if(type == "semantic"){
                    // ─── Rollenbasiertes Crossover: Homologie aus dem Knoteninhalt ──────────
                    //
                    // Warum ueberhaupt: "uniform"/"onepoint" paaren nach ARRAY-POSITION, und
                    // "innovation" nach HERKUNFT (Node::innovationID). Beides ist fuer GNP
                    // schwach begruendet. Die Netze starten klein und wachsen/schrumpfen ueber
                    // addDelNodes(); jede Loeschung verschiebt alle nachfolgenden Indizes, also
                    // bezeichnet dieselbe Position in zwei Individuen nach einigen hundert
                    // Generationen nichts Gemeinsames mehr. Herkunftsmarker wiederum verhindern
                    // zwar, dass "competing conventions" NEU entstehen, koennen aber KONVERGENTE
                    // Loesungen nicht erkennen: zwei Linien, die unabhaengig voneinander dieselbe
                    // Entscheidungsregel entwickeln, tragen verschiedene IDs und gelten als
                    // verschiedene Gene.
                    //
                    // Ein GNP-Knoten braucht das alles nicht -- er traegt seine Rolle explizit:
                    //   type            "J"/"JE" (Urteil) oder "P" (Aktion)
                    //   f               Feature-Index (J) bzw. Aktion (P)
                    //   edges.size()    Stelligkeit: in wie viele Intervalle geteilt wird
                    //   boundaries      wo genau geschnitten wird (boundaries.size() == edges+1)
                    // (type, f, edges.size()) ist die grobe Rolle, boundaries die Feinjustierung
                    // darin. Die Stelligkeit gehoert zwingend in die Rolle: sonst wuerde ein
                    // Austausch die Kantenzahl des Empfaengers aendern, und die Schwellenvektoren
                    // waeren nicht vergleichbar.
                    //
                    // Ablauf: (1) beide Eltern nach der Rolle gruppieren, (2) innerhalb einer
                    // Gruppe ueber den Schwellenabstand paaren, (3) je Paar mit `propability`
                    // tauschen, (4) die Kanten des uebernommenen Knotens ueber die Paarungs-
                    // tabelle uebersetzen. Nicht zuordenbare Knoten bleiben, wo sie sind -- im
                    // Gegensatz zu "innovation" wird NICHTS angehaengt, dieses Crossover laesst
                    // das Netz also nicht wachsen.
                    if(!distributionBernoulli(*generator)) continue; // Gate PRO PAAR (= propability)

                    auto meanFitnessOf = [](const Network& net) -> float {
                        if(net.fitnessValues.empty()) return net.fitness;
                        float sum = 0.0f;
                        for(float v : net.fitnessValues) sum += v;
                        return sum / static_cast<float>(net.fitnessValues.size());
                    };

                    // Elite gibt nur ab und empfaengt nie; sonst spendet der fittere Elternteil
                    // und bleibt unveraendert -- wie bei "innovation".
                    Network* recipient = nullptr;
                    Network* donorNet  = nullptr;
                    if(parent1IsElite){
                        donorNet = &parent1; recipient = &parent2;
                    } else if(parent2IsElite){
                        donorNet = &parent2; recipient = &parent1;
                    } else if(meanFitnessOf(parent1) >= meanFitnessOf(parent2)){
                        donorNet = &parent1; recipient = &parent2;
                    } else {
                        donorNet = &parent2; recipient = &parent1;
                    }

                    // Rollenschluessel eines Knotens. Als String, damit er direkt als Map-Key
                    // taugt; die Gruppen sind klein, das ist nicht performancekritisch.
                    auto roleKey = [](const Node& node) -> std::string {
                        return node.type + "|" + std::to_string(node.f)
                                         + "|" + std::to_string(node.edges.size());
                    };

                    // Abstand zweier Knoten derselben Rolle: mittlere, auf den Feature-Bereich
                    // normierte Abweichung der INNEREN Schwellen. Der erste und letzte Wert von
                    // boundaries sind Unter- und Obergrenze des Features und tragen keine
                    // Information ueber die Entscheidungsregel. Die Normierung sorgt dafuer, dass
                    // ein Feature mit Spanne 20 nicht staerker gewichtet wird als eines mit
                    // Spanne 5. P-Knoten haben keine Schwellen -> Abstand 0, sie sind innerhalb
                    // ihrer Rolle austauschbar.
                    auto boundaryDistance = [](const Node& a, const Node& b) -> float {
                        if(a.boundaries.size() < 3 || a.boundaries.size() != b.boundaries.size()){
                            return 0.0f;
                        }
                        double rangeA = a.boundaries.back() - a.boundaries.front();
                        double rangeB = b.boundaries.back() - b.boundaries.front();
                        double range = std::max(1e-9, 0.5 * (rangeA + rangeB));
                        double sum = 0.0;
                        size_t n = 0;
                        for(size_t k = 1; k + 1 < a.boundaries.size(); k++){
                            sum += std::abs(a.boundaries[k] - b.boundaries[k]) / range;
                            n++;
                        }
                        return (n == 0) ? 0.0f : static_cast<float>(sum / static_cast<double>(n));
                    };

                    // Kandidaten je Rolle einsammeln. matchOnlyUsed blendet Knoten aus, die in
                    // der letzten Auswertung nie betreten wurden -- dann wird keine tote Masse
                    // getauscht (siehe der hohe Anteil dormanter Knoten im Wachstumsplot).
                    std::unordered_map<std::string, std::vector<int>> donorByRole, recipientByRole;
                    for(size_t k = 0; k < donorNet->innerNodes.size(); k++){
                        if(matchOnlyUsed && !donorNet->innerNodes[k].used) continue;
                        donorByRole[roleKey(donorNet->innerNodes[k])].push_back(static_cast<int>(k));
                    }
                    for(size_t k = 0; k < recipient->innerNodes.size(); k++){
                        if(matchOnlyUsed && !recipient->innerNodes[k].used) continue;
                        recipientByRole[roleKey(recipient->innerNodes[k])].push_back(static_cast<int>(k));
                    }

                    // Paarungstabelle Spender-Index -> Empfaenger-Index. Innerhalb einer Rolle
                    // wird global gierig zugeordnet: alle Kreuzpaare nach Abstand sortieren und
                    // von vorn nehmen, solange beide Partner noch frei sind. Eine optimale
                    // Zuordnung (ungarische Methode) waere theoretisch sauberer, aber die Gruppen
                    // umfassen typischerweise nur wenige Knoten, und dort stimmen beide Verfahren
                    // fast immer ueberein -- der Aufwand lohnt hier nicht.
                    std::unordered_map<int, int> donorToRecipient;
                    for(auto const& [role, donorIdxs] : donorByRole){
                        auto it = recipientByRole.find(role);
                        if(it == recipientByRole.end()) continue; // Rolle nur beim Spender -> disjunkt
                        const std::vector<int>& recipientIdxs = it->second;

                        struct Candidate { float distance; int donorIdx; int recipientIdx; };
                        std::vector<Candidate> candidates;
                        candidates.reserve(donorIdxs.size() * recipientIdxs.size());
                        for(int d : donorIdxs){
                            for(int r : recipientIdxs){
                                float distance = boundaryDistance(donorNet->innerNodes[d],
                                                                  recipient->innerNodes[r]);
                                // Gleiche Rolle, aber zu unterschiedliche Schwellen: die beiden
                                // schneiden das Feature an ganz verschiedenen Stellen und sind
                                // dann eben NICHT dasselbe Gen.
                                if(distance > boundaryTolerance) continue;
                                candidates.push_back({distance, d, r});
                            }
                        }
                        std::sort(candidates.begin(), candidates.end(),
                                  [](const Candidate& a, const Candidate& b){ return a.distance < b.distance; });

                        std::unordered_set<int> donorTaken, recipientTaken;
                        for(const Candidate& c : candidates){
                            if(donorTaken.count(c.donorIdx) || recipientTaken.count(c.recipientIdx)) continue;
                            donorTaken.insert(c.donorIdx);
                            recipientTaken.insert(c.recipientIdx);
                            donorToRecipient.emplace(c.donorIdx, c.recipientIdx);
                        }
                    }

                    if(donorToRecipient.empty()) continue; // keine gemeinsame Rolle -> Paar unveraendert
                    crossoverNodesMatched += static_cast<int>(donorToRecipient.size());

                    // Austausch, ALLES-ODER-NICHTS pro Knoten.
                    //
                    // Bei einem Judgment-Knoten gehoeren `boundaries` und `edges` zusammen: Kante k
                    // heisst "faellt das Feature in Intervall k, gehe dorthin". Uebernaehme man die
                    // Schwellen des Spenders, liesse aber einzelne Kanten des Empfaengers stehen,
                    // entstuende ein Knoten mit Spender-Intervallen und teils Empfaenger-Zielen --
                    // eine Kombination, die KEIN Elternteil je getestet hat. Deshalb wird ein Knoten
                    // nur dann uebernommen, wenn sich JEDE seiner Kanten uebersetzen laesst.
                    //
                    // Das hat eine Kettenwirkung, die boundaryTolerance zum eigentlichen Regler
                    // macht: die Paarungstabelle enthaelt nur Paare INNERHALB der Toleranz, also ist
                    // eine Kante auf einen nicht gepaarten Knoten nicht uebersetzbar. Ein Gen wandert
                    // damit nur, wenn es UND alle seine direkten Nachfolger eine nahe Entsprechung
                    // haben -- es wandert immer ein vollstaendiges, im Spender erprobtes Teilstueck.
                    //
                    // Zwei getrennte Raten: `propability` entscheidet weiter oben, OB ein Paar
                    // ueberhaupt rekombiniert, `nodeExchangeRate` hier, WIE VIEL innerhalb eines
                    // ausgewaehlten Paares wandert. Das ist Absicht -- ein Paar rekombiniert
                    // entweder spuerbar oder gar nicht, statt ueber die ganze Population duenn
                    // zu streuen.
                    bool changed = false;
                    for(auto const& [donorIdx, recipientIdx] : donorToRecipient){
                        if(!distributionNodeExchange(*generator)) continue;

                        const Node& donorNode = donorNet->innerNodes[donorIdx];

                        // Erst uebersetzen, dann entscheiden -- der Empfaengerknoten wird nicht
                        // angefasst, solange nicht feststeht, dass das ganze Gen passt.
                        std::vector<int> translatedEdges;
                        translatedEdges.reserve(donorNode.edges.size());
                        bool translatable = true;
                        for(int donorTarget : donorNode.edges){
                            auto mapped = donorToRecipient.find(donorTarget);
                            if(mapped == donorToRecipient.end()){
                                translatable = false;
                                break;
                            }
                            translatedEdges.push_back(mapped->second);
                        }
                        if(!translatable){
                            crossoverNodesSkipped++;
                            continue;
                        }

                        Node& targetNode = recipient->innerNodes[recipientIdx];
                        unsigned int keptPosition = targetNode.id;
                        targetNode = donorNode;
                        targetNode.id = keptPosition;              // die Position gehoert dem Empfaenger
                        targetNode.edges = std::move(translatedEdges);
                        targetNode.generationReceived = currentGeneration;
                        targetNode.traverseCounter = 0;
                        targetNode.lastVisitStep = 0;
                        crossoverNodesExchanged++;
                        changed = true;
                    }

                    if(!changed) continue;
                    crossoverPairsApplied++;

                    recipient->changeFalseEdges();
                    recipient->nCrossovers += 1;
                    recipient->structureChangedThisGen = true;
                    recipient->blendLineageWith(*donorNet, lineagePriorCap);
                    continue;

                } else if(type == "uniform"){
                    nNodesToExchange = std::min(parent1.innerNodes.size(), parent2.innerNodes.size());
                    // set nodesToExchange
                    for(int k=0; k<nNodesToExchange; k++){
                        bool result = distributionBernoulli(*generator);
                        if(result == true){
                            nodesToExchange.push_back(k);
                        }
                    }

                } else if (type == "onepoint") {
                    // with probability (1 - propability) this pair is left untouched. Once the pair
                    // is selected, the cutpoint itself stays uniformly drawn.
                    if(!distributionBernoulli(*generator)){
                        continue; // onepoint crossover should not be applied, skip to next pair
                    }

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

                    int maxNodesToExchange = std::max(parent1.innerNodes.size(), parent2.innerNodes.size());
                    std::uniform_int_distribution<int> distributionUniform(2, maxNodesToExchange);
                    int depthI = distributionUniform(*generator);
                    int depthII = distributionUniform(*generator);

                    std::vector<int> successor1 = findSuccessorNodes(
                            parent1, 
                            -1, 
                            -1, 
                            traversalNeighbor, 
                            depthI,
                            lowerBoundTraversalCounter, 
                            upperBoundTraversalCounter
                            ); // getting the node indices of subnetwork
                    std::vector<int> successor2 = findSuccessorNodes(
                            parent2, 
                            -1, 
                            -1, 
                            traversalNeighbor, 
                            depthII, 
                            lowerBoundTraversalCounter, 
                            upperBoundTraversalCounter
                            ); // getting the node indices of subnetwork

                    // prevent networks <= 2 inner nodes otherwise the crossover would delete to many nodes
                    if((int)successor1.size() - (int)successor2.size() >= (int)parent1.innerNodes.size() - 2 ||
                       (int)successor2.size() - (int)successor1.size() >= (int)parent2.innerNodes.size() - 2 ||
                       (successor1.size() == 1 && successor2.size() == 1) ) {
                        continue;
                    }

                    // prevent crossover of unused AND protected nodes
                    if(successor1.size() == 1 && 
                            currentGeneration - parent1.innerNodes[successor1[0]].generationReceived < crossoverProtection) continue;

                    if(successor2.size() == 1 && 
                            currentGeneration - parent2.innerNodes[successor2[0]].generationReceived < crossoverProtection) continue;


                    if(parent1IsElite && successor1.size() > 1){
                        // parent1 is elite → only parent2 receives genes
                        parent2.nCrossovers += 1;
                        parent2.structureChangedThisGen = true;
                        // Elite bleibt selbst unveraendert; nur der Empfaenger (parent2)
                        // vermischt seine Lineage-Statistik mit der des Elite-Spenders.
                        parent2.blendLineageWith(parent1, lineagePriorCap);

                        int minSubNodes = std::min(successor1.size(), successor2.size());

                        std::unordered_map<int, int> swapMap1 = initNodeSwapMap(successor1, successor2, parent2.innerNodes.size());

                        // Copy nodes from parent1 to parent2
                        for(int j=0; j<minSubNodes; j++){
                            parent2.innerNodes[successor2[j]] = parent1.innerNodes[successor1[j]];
                            parent2.innerNodes[successor2[j]].generationReceived = currentGeneration;
                        }

                        std::vector<int> indices1;
                        indices1.reserve(swapMap1.size()); 
                        for (auto const& [key, val] : swapMap1) {
                            indices1.push_back(val);
                        }

                        if(successor1.size() > successor2.size()){
                            addOverhangNodes(successor1, successor2, parent1, parent2, true, currentGeneration);
                        }

                        parent2.remapNodeIdsAndEdges(swapMap1, indices1, false);
                        
                        if (successor1.size() < successor2.size()) {
                            deleteOverhangNodes(successor2, successor1, parent2);
                        }

                        parent2.changeFalseEdges();

                    } else if(parent2IsElite && successor2.size() > 1){
                        // parent2 is elite → only parent1 receives genes
                        parent1.nCrossovers += 1;
                        parent1.structureChangedThisGen = true;
                        // Elite bleibt selbst unveraendert; nur der Empfaenger (parent1)
                        // vermischt seine Lineage-Statistik mit der des Elite-Spenders.
                        parent1.blendLineageWith(parent2, lineagePriorCap);

                        int minSubNodes = std::min(successor1.size(), successor2.size());

                        std::unordered_map<int, int> swapMap2 = initNodeSwapMap(successor2, successor1, parent1.innerNodes.size());

                        // Copy nodes from parent2 to parent1
                        for(int j=0; j<minSubNodes; j++){
                            parent1.innerNodes[successor1[j]] = parent2.innerNodes[successor2[j]];
                            parent2.innerNodes[successor2[j]].generationReceived = currentGeneration;
                        }

                        std::vector<int> indices2;
                        indices2.reserve(swapMap2.size()); 
                        for (auto const& [key, val] : swapMap2) {
                            indices2.push_back(val);
                        }

                        if(successor2.size() > successor1.size()){
                           addOverhangNodes(successor2, successor1, parent2, parent1, true, currentGeneration);
                        }

                        parent1.remapNodeIdsAndEdges(swapMap2, indices2, false);

                        if (successor2.size() < successor1.size()) {
                            deleteOverhangNodes(successor1, successor2, parent1);
                        }
                       
                        parent1.changeFalseEdges();

                    } else { // no elite --> bidirectional swap

                        // swap map from individual1 to individual2
                        std::unordered_map<int, int> swapMap1 = initNodeSwapMap(successor1, successor2, parent2.innerNodes.size());
                        // swap map from individual2 to individual1
                        std::unordered_map<int, int> swapMap2 = initNodeSwapMap(successor2, successor1, parent1.innerNodes.size()); 

                        parent1.nCrossovers += 1;
                        parent2.nCrossovers += 1;
                        parent1.structureChangedThisGen = true;
                        parent2.structureChangedThisGen = true;
                        blendLineageBidirectional(parent1, parent2);

                        // exchange all nodes until same subnetwork size is reached 
                        int minSubNodes = std::min(successor1.size(), successor2.size());
                        for(int j=0; j<minSubNodes; j++){ 
                            std::swap(parent1.innerNodes[successor1[j]], parent2.innerNodes[successor2[j]]);
                            parent1.innerNodes[successor1[j]].generationReceived = currentGeneration;
                            parent2.innerNodes[successor2[j]].generationReceived = currentGeneration;
                        }

                        // add overhang nodes
                        if(successor1.size() > successor2.size()){
                           addOverhangNodes(successor1, successor2, parent1, parent2, false, currentGeneration);
                        } else if (successor1.size() < successor2.size()) {
                           addOverhangNodes(successor2, successor1, parent2, parent1, false, currentGeneration);
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

                    // Lineage-Statistik entsprechend der tatsaechlichen Genflussrichtung
                    // vermischen (einmal pro Paar, nicht pro ausgetauschtem Knoten).
                    // nCrossovers wird ebenfalls hier (einmal pro Paar) erhoeht, analog zum
                    // "randomWidth"-Zweig, damit die Analyse-Zaehlung fuer alle Crossover-Typen konsistent ist.
                    if(!nodesToExchange.empty()){
                        if(parent1IsElite){
                            parent2.blendLineageWith(parent1, lineagePriorCap);
                            parent2.nCrossovers += 1;
                        } else if(parent2IsElite){
                            parent1.blendLineageWith(parent2, lineagePriorCap);
                            parent1.nCrossovers += 1;
                        } else {
                            blendLineageBidirectional(parent1, parent2);
                            parent1.nCrossovers += 1;
                            parent2.nCrossovers += 1;
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
                Network& parent2,
                bool copy = false,
                int currentGeneration = 0
                ){

            int overhang = successor1.size() - successor2.size();
            for(int i=0; i<overhang; i++){
                int nodeIndex = successor1[successor2.size()+i];
                if(copy == false){
                    parent2.innerNodes.push_back(std::move(parent1.innerNodes[nodeIndex]));
                } else {
                    parent2.innerNodes.push_back(parent1.innerNodes[nodeIndex]);
                }
                parent2.innerNodes.back().generationReceived = currentGeneration;
                // Keep parent2's coactivation matrix in sync so that
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
                map[nodeIndex] = parent1.innerNodes[0].changeEdge(parent1.innerNodes.size(), nodeIndex, &parent1.innerNodes);
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
         * @param traversalNeighbor If true, collects nodes with traverse counters within ±10% of the start node's counter, instead of strictly greater.
         * @return A sorted vector of node indices comprising the start node and up to
         *         @p nSelectedNodes successors.
         */
        std::vector<int> findSuccessorNodes(
                auto& individual, 
                int subNodesStart = -1, 
                int nSelectedNodes = -1, 
                bool traversalNeighbor = false, 
                int graphDepth = -1,
                float lowerBoundTraversalCounter = 0.9,
                float upperBoundTraversalCounter = 1.1
                ){

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
                // If the node is frozen, all nodes with the same frozen value are considered successors.
                if (individual.innerNodes[subNodesStart].frozen > 0){
                    for(auto& node : individual.innerNodes){
                        if (individual.innerNodes[subNodesStart].frozen == node.frozen){
                            nodeIndices.push_back(node.id);
                        }
                    }
                }

                return nodeIndices; // if the node is unused, no successor nodes can be found
            }

            nodeIndices.push_back(subNodesStart);

            //std::uniform_int_distribution<int> distributionUniformII(2, 10);
            //graphDepth = distributionUniformII(*generator);
            // Graph-depth-based BFS traversal through actual node edges
            if (graphDepth != -1) {
                std::unordered_set<int> visited;
                std::queue<std::pair<int, int>> bfsQueue; // {node_index, current_depth}
                bfsQueue.push({subNodesStart, 0});
                visited.insert(subNodesStart);

                while (!bfsQueue.empty()) {
                    auto [nodeIdx, currentDepth] = bfsQueue.front();
                    bfsQueue.pop();

                    if (nodeIdx != subNodesStart) {
                        nodeIndices.push_back(nodeIdx);
                        // respect nSelectedNodes as an upper bound (start node excluded from count)
                        if ((int)nodeIndices.size() - 1 >= nSelectedNodes) {
                            break;
                        }
                    }

                    if (currentDepth < graphDepth) {
                        for (int edgeTarget : individual.innerNodes[nodeIdx].edges) {
                            if (visited.find(edgeTarget) == visited.end()) {
                                visited.insert(edgeTarget);
                                bfsQueue.push({edgeTarget, currentDepth + 1});
                            }
                        }
                    }
                }
                std::sort(nodeIndices.begin(), nodeIndices.end());
                return nodeIndices;
            }


            // Zwei getrennte Kriterien, zwei getrennte Felder (siehe Node::traverseCounter /
            // Node::lastVisitStep): die Reihenfolge entlang des Pfades ("nach dem Startknoten
            // besucht") kommt aus dem Zeitstempel lastVisitStep, die Nutzungs-Aehnlichkeit
            // ("etwa gleich oft besucht") aus der Haeufigkeit traverseCounter. Vor der Trennung
            // trug traverseCounter in decisionAndNextNode() den Zeitstempel, weshalb das
            // Verhaeltnis-Kriterium unten faktisch Zeitstempel verglichen hat.
            int lastVisitStepStart = individual.innerNodes[subNodesStart].lastVisitStep;
            int traverseCounterStart = individual.innerNodes[subNodesStart].traverseCounter;
            int lastFrozenValue = individual.innerNodes[subNodesStart].frozen;
            for(int i=0; i<individual.innerNodes.size(); i++){
                int lastVisitStepNode = individual.innerNodes[i].lastVisitStep;
                int traverseCounterNode = individual.innerNodes[i].traverseCounter;
                int frozenValueNode = individual.innerNodes[i].frozen;
                // Die frozen-Klausel haelt eingefrorene Bloecke zusammen und darf nur fuer
                // TATSAECHLICH eingefrorene Knoten (frozen > 0) greifen. Ohne das
                // frozen > 0 ist sie bei einem Netz ohne eingefrorene Knoten (frozen
                // ueberall 0, der Normalfall) IMMER wahr -- dann liefert
                // findSuccessorNodes() unabhaengig vom Pfad einfach die ersten
                // nSelectedNodes Knoten, und "randomWidth" tauscht einen beliebigen
                // Indexbereich statt eines Sub-Netzes.
                bool sameFrozenBlock = (frozenValueNode > 0 && frozenValueNode == lastFrozenValue);

                if(traversalNeighbor == false){
                    if(lastVisitStepNode > lastVisitStepStart || sameFrozenBlock){
                        nodeIndices.push_back(i);
                        nSelectedNodes --;
                        if (nSelectedNodes == 0){
                            break;
                        }
                    }
                } else {
                    if(sameFrozenBlock ||
                            (traverseCounterStart * lowerBoundTraversalCounter <= traverseCounterNode && traverseCounterNode <= traverseCounterStart * upperBoundTraversalCounter)){
                        if(i != subNodesStart){ // exclude the start node itself from being considered a neighbor
                            nodeIndices.push_back(i);
                        }
                    }
                }
                lastFrozenValue = frozenValueNode;
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
         * @param junk ratio of protected unused nodes (junk DNA). A value of 0.1 protects 10% of unused nodes 
         * and at least one node is always protected. 
         * @param noElite If true, elite individuals are protected 
         *  normaly this is not necessary because the operator addDelNodes() is fitness neutral. But because of 
         *  the fitness neutraly is given by used nodes it just protects node of the last traversal path. 
         *  If you use multiple traversal path per generation, e.g. using multiple seeds for evaluation a elite protection is not garanteed and 
         *  noElite should be set to true. Default is false.
         * 
         * @note This operator has not influence on the individuals fitness  
         * @see Network::addDelNodes()
         */
        void callAddDelNodes(
                std::vector<float>& minF, 
                std::vector<float>& maxF, 
                float junk=0, 
                bool noElite = false,
                int currentGeneration = 0,
                int crossoverProtection = 3,
                int nodeGracePeriod = -1
                ){

            for(int i=0; i<individuals.size(); i++){

                if (noElite && std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end()) {continue;}// skip elite individuals if noElite is true

                if(individuals[i].addDelNodes(minF, maxF, junk, nFeatureValues, currentGeneration, crossoverProtection, nodeGracePeriod)){
                    individuals[i].structureChangedThisGen = true;
                }

            }
        }

        /**
         * @brief Advances the grace-period counters of every individual by one generation.
         *
         * @details
         * Call once per generation between the evaluation and callAddDelNodes(), while
         * Node::used still reflects the generation that just finished.
         *
         * @see Network::ageUnusedNodes()
         */
        void callAgeUnusedNodes(){
            for(auto& network : individuals){
                network.ageUnusedNodes();
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
         * @param survivalMode See Network::fitGymnasium(). Passed through unchanged so
         *        gymnasium()/renderVideos() (video path) always matches the reward mode
         *        used here during training.
         * @param potential See Network::fitGymnasium().
         * @param landingQuote If true, aggregates fitness as
         *        Return * (landings / nSeeds)^k instead of the plain mean reward.
         *        This scales (rather than additively combines) the raw return with
         *        the landing rate, so an individual that lands on every seed but
         *        flies conservatively (e.g. 120 * 1.0^2 = 120) can outrank one that
         *        posts higher raw scores but crashes on some seeds (e.g. 8/10 landings,
         *        160 * 0.8^2 = 102.4), while a "suicidal" high-scorer that rarely lands
         *        (5/10, 200 * 0.5^2 = 50.0) is penalized heavily. Default: false (mean
         *        reward, unchanged behavior).
         * @param landingQuoteExponent The exponent k in (landings/nSeeds)^k, expected
         *        in [2,3]. Only used when landingQuote is true. Default: 2.0f.
         * 
         */
        void gymnasiumMultiSeed(
            GymEnvWrapper& env,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            const std::vector<int>& seeds,
            bool validation = false,
            float curriculumLevel = 1.0f,
            bool absoluteImpulseCurriculum = false,
            bool useLineageFitness = true,
            bool uniformDirectionCurriculum = false,
            const std::vector<float>& directionAngles = {},
            bool survivalMode = false,
            bool potential = false,
            bool landingQuote = false,
            float landingQuoteExponent = 2.0f
                ){

            for(auto& network : individuals){
                network.fitnessValues.clear();
                network.lastStepRewards.clear();
                network.lastStepRewardsII.clear();
                network.episodeLog.clear();
                network.visitedNodesPerSeed.clear();
                network.objectivesPerSeed.clear();
                float totalReward = 0.0f;
                float minReward = std::numeric_limits<float>::max();
                int landingCount = 0;
                bool firstSeed = true;
                // Akkumulatoren fuer den ueber alle Seeds gemittelten Epsilon-Lexicase
                // Ziel-Vektor (siehe Network::lastEpisodeObjectives / lexicaseObjectives).
                std::vector<float> objSum(Network::N_OBJECTIVES, 0.0f);

                for(size_t seedIdx = 0; seedIdx < seeds.size(); ++seedIdx){
                    int s = seeds[seedIdx];
                    // directionAngles[i] enthaelt den Winkel, der Seed seeds[i] fuer
                    // diese Generation zugewiesen wurde (siehe drawSeeds()/lunarlander.py)
                    // -- garantiert eine gleichverteilte Abdeckung aller Richtungen ueber
                    // das Batch hinweg, statt der natuerlichen (quadratisch verteilten)
                    // Seed-Richtung.
                    float directionAngle = (uniformDirectionCurriculum && seedIdx < directionAngles.size())
                        ? directionAngles[seedIdx] : 0.0f;

                    // Snapshot der traverseCounter VOR diesem Seed, um im Anschluss per Diff genau
                    // die fuer DIESEN Seed durchlaufenen Knoten zu isolieren (traverseCounter/used
                    // akkumulieren sonst absichtlich ueber alle Seeds der Generation hinweg, siehe
                    // Network::visitedNodesPerSeed). Rein additiv/lesend -- aendert das bestehende
                    // Akkumulationsverhalten von used/traverseCounter nicht.
                    std::vector<unsigned int> traverseCounterBefore(network.innerNodes.size());
                    for(size_t n = 0; n < network.innerNodes.size(); ++n){
                        traverseCounterBefore[n] = network.innerNodes[n].traverseCounter;
                    }

                    if(firstSeed == true){
                        network.fitGymnasium(env, dMax, maxSteps, maxConsecutiveP, worstFitness, s, true, validation, false, curriculumLevel, absoluteImpulseCurriculum, uniformDirectionCurriculum, directionAngle, survivalMode, potential);
                    } else {
                        network.fitGymnasium(env, dMax, maxSteps, maxConsecutiveP, worstFitness, s, false, validation, false, curriculumLevel, absoluteImpulseCurriculum, uniformDirectionCurriculum, directionAngle, survivalMode, potential);
                    }

                    std::vector<int> visitedThisSeed;
                    for(size_t n = 0; n < network.innerNodes.size(); ++n){
                        if(network.innerNodes[n].traverseCounter != traverseCounterBefore[n]){
                            visitedThisSeed.push_back(static_cast<int>(n));
                        }
                    }
                    network.visitedNodesPerSeed.push_back(std::move(visitedThisSeed));

                    //if (!network.frozenExperience)
                      //  network.updateExperienceFromEpisode();
                    //else
                      //  network.episodeLog.clear();

                    network.fitnessValues.push_back(network.fitness);
                    network.lastStepRewards.push_back(network.lastFitness);
                    network.lastStepRewardsII.push_back(network.lastFitnessII);
                    for(size_t o = 0; o < objSum.size() && o < network.lastEpisodeObjectives.size(); o++){
                        objSum[o] += network.lastEpisodeObjectives[o];
                    }
                    // Keep the per-seed vector as well: averaging it away (lexicaseObjectives)
                    // costs exactly the seed-by-seed resolution lexicase lives on.
                    network.objectivesPerSeed.push_back(network.lastEpisodeObjectives);
                    bool landedThisSeed = network.fitness > Network::LANDING_SUCCESS_THRESHOLD;
                    if(landedThisSeed){
                        landingCount++;
                    }
                    if(useLineageFitness){
                        network.updateLineageStats(network.fitness, landedThisSeed);
                    }
                    totalReward += network.fitness;
                    firstSeed = false;

                    if(network.fitness < minReward){
                        minReward = network.fitness;
                    }
                }
                
                //network.updateExperienceFromEpisode();

                // Ueber alle Seeds gemittelter 5-D Ziel-Vektor fuer Population::lexicaseSelection().
                network.lexicaseObjectives.assign(objSum.size(), 0.0f);
                for(size_t o = 0; o < objSum.size(); o++){
                    network.lexicaseObjectives[o] = objSum[o] / static_cast<float>(seeds.size());
                }

                // Default aggregation: mean reward of the current generation's seed batch
                float rawFitness = totalReward / static_cast<float>(seeds.size());
                if(landingQuote){
                    // Fitness_gesamt = Return * (Landeanzahl / N_seeds)^k
                    // Skaliert (statt addiert) den Return mit der Landequote, damit
                    // konservative "immer landet" Individuen hohe aggressive Scorer mit
                    // vielen Crashes ausstechen koennen (siehe Docstring-Rechenbeispiel).
                    float landingRate = static_cast<float>(landingCount) / static_cast<float>(seeds.size());
                    // network.fitness = rawFitness * std::pow(landingRate, landingQuoteExponent);
                    network.fitness = landingRate;
                } else {
                    network.fitness = rawFitness;
                }
                // network.fitness = minReward;
                // float rawFitness = minReward;

                // EMA über Generationen hinweg: network.fitness bleibt bis zur Zuweisung
                // unangetastet, damit die vorige EMA-Historie (network.emaFitness) beim
                // Update noch verfuegbar ist. network.alpha ist ein individuelles Member
                // (spaeter aehnlichkeitsbasiert bei Mutation/Crossover anpassbar).
                // if (!validation) {
                //     if (!network.emaInitialized) {
                //         network.emaFitness = rawFitness;  // Erstinitialisierung
                //         network.emaInitialized = true;
                //     } else {
                //         network.emaFitness = network.alpha * rawFitness
                //                             + (1.0f - network.alpha) * network.emaFitness;
                //     }
                //     network.fitness = network.emaFitness;
                // } else {
                //     network.fitness = rawFitness;  // Validierung: kein EMA
                // }
                //
            }
        }

        /**
         * @brief Evaluates all individuals across multiple seeds in parallel using N cores.
         *
         * @details
         * Parallelized version of gymnasiumMultiSeed(). Each core receives its own
         * environment instance to avoid thread-safety issues.  The population is split
         * into roughly equal chunks and each chunk is evaluated in its own thread.
         *
         * Because fitGymnasium() calls Python (env.reset / env.step), each worker
         * thread acquires the GIL before evaluating an individual and releases it
         * afterwards, allowing other threads to interleave.  For Gymnasium
         * environments backed by C extensions (MuJoCo, Atari, Box2D …), the
         * extension typically releases the GIL during step(), enabling true parallel
         * execution.
         *
         * @param envs  Vector of GymEnvWrapper objects, one per core.  envs.size()
         *              determines the number of threads.
         * @param dMax  Maximum consecutive judgment nodes per decision
         * @param maxSteps Maximum episode length
         * @param maxConsecutiveP Maximum consecutive processing nodes allowed
         * @param worstFitness Fitness value assigned when networks violate constraints
         * @param seeds Vector of random seeds for environment initialization
         */

        /*
        void gymnasiumMultiSeed(
            std::vector<GymEnvWrapper>& envs,
            int dMax,
            int maxSteps,
            int maxConsecutiveP,
            int worstFitness,
            const std::vector<int>& seeds,
            bool useLineageFitness = true
                ){

            int nCores = static_cast<int>(envs.size());
            int nInd   = static_cast<int>(individuals.size());

            // Worker lambda: evaluate individuals[start..end) using envs[threadIdx].
            // The GIL is acquired only around fitGymnasium (which calls Python via
            // env.reset / env.step) and released for the surrounding bookkeeping
            // (vector ops, arithmetic) to maximise parallel overlap.
            auto worker = [&](int start, int end, int threadIdx) {
                for (int i = start; i < end; i++) {
                    auto& network = individuals[i];
                    network.fitnessValues.clear();
                    network.lastStepRewards.clear();
                    float totalReward = 0.0f;
                    bool firstSeed = true;

                    for (int s : seeds) {
                        {
                            py::gil_scoped_acquire acquire;
                            network.fitGymnasium(
                                envs[threadIdx], dMax, maxSteps,
                                maxConsecutiveP, worstFitness, s, firstSeed);
                        }
                        network.fitnessValues.push_back(network.fitness);
                        network.lastStepRewards.push_back(network.lastFitness);
                        if(useLineageFitness){
                            network.updateLineageStats(network.fitness, network.fitness > Network::LANDING_SUCCESS_THRESHOLD);
                        }
                        totalReward += network.fitness;
                        firstSeed = false;
                    }

                    network.fitness = totalReward / static_cast<float>(seeds.size());
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(nCores);

            for (int t = 0; t < nCores; t++) {
                int start = t * nInd / nCores;
                int end   = (t + 1) * nInd / nCores;
                if (start == end) continue;          // skip empty chunks
                threads.emplace_back(worker, start, end, t);
            }

            for (auto& th : threads) {
                th.join();
            }
        }
        */

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
                float minReward = std::numeric_limits<float>::max();
                float maxReward = std::numeric_limits<float>::min();
                float totalVelocity = 0.0f;
                float totalAbsX   = 0.0f;

                for(size_t i = 0; i < network.fitnessValues.size(); i++){
                    totalReward += network.fitnessValues[i];
                    totalVelocity += network.lastStepRewards[i] * -1;
                    totalAbsX   += network.lastStepRewardsII[i] * -1;
                    if(network.fitnessValues[i] < minReward){
                        minReward = network.fitnessValues[i];
                    }
                    if(network.fitnessValues[i] > maxReward){
                        maxReward = network.fitnessValues[i];
                    }

                }
                
                // Median Reward berechnen
                std::vector<float> sortedRewards = network.fitnessValues;
                std::sort(sortedRewards.begin(), sortedRewards.end());
                float medianReward;

                size_t n = sortedRewards.size();
                if(n % 2 == 0){
                    medianReward = (sortedRewards[n / 2 - 1] + sortedRewards[n / 2]) / 2.0f;
                } else {
                    medianReward = sortedRewards[n / 2];
                }
                     
                float meanReward = totalReward / static_cast<float>(network.fitnessValues.size());
                float meanVelocity = totalVelocity / static_cast<float>(network.fitnessValues.size());
                float meanAbsX   = totalAbsX / static_cast<float>(network.fitnessValues.size());
                float diffReward = maxReward - minReward;

                network.objectives = {meanReward, minReward};
                network.fitness = meanReward;
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
         * @brief Performs Pareto-based tournament selection with Utopia-point elitism.
         *
         * @details
         * Extends paretoTournamentSelection with a single elite group:
         * The E best individuals are selected by their Euclidean distance
         * to the Utopia point (the vector of per-objective maxima across
         * the entire population). This preserves individuals that are
         * well-rounded across all objectives.
         *
         * @param N Tournament size
         * @param E Number of elite individuals by Utopia distance
         */
        void paretoTournamentSelection(int N, int E){
            std::vector<Network> selection;
            selection.reserve(individuals.size());
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
            meanFitness = 0;
            minFitness = individuals[0].fitness;
            bestFit = individuals[0].fitness;
            maxNetworkSize = 0;

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

            // Utopia-point elitism
            setEliteUtopia(E, individuals, selection);
            individuals = std::move(selection);
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
            meanFitness /= individuals.size();
        }

        /**
         * @brief Identifies elite individuals by Euclidean distance to the Utopia point.
         *
         * @details
         * The Utopia point is constructed as the per-objective maximum across
         * the entire population. Each individual's distance to this point is
         * computed (objectives are normalized by range to avoid scale bias).
         * The E closest individuals are added to the selection as elites.
         *
         * @param E Number of elite individuals
         * @param individuals Current population (const reference)
         * @param selection New population being constructed (output)
         */
        void setEliteUtopia(
                int E,
                std::vector<Network>& individuals,
                std::vector<Network>& selection){

            indicesElite.clear();
            if(E <= 0 || individuals.empty()) return;

            size_t nObj = individuals[0].objectives.size();

            // Compute Utopia point (per-objective max) and nadir (per-objective min)
            std::vector<float> utopia(nObj, std::numeric_limits<float>::lowest());
            std::vector<float> nadir(nObj, std::numeric_limits<float>::max());
            for(const auto& ind : individuals){
                for(size_t o = 0; o < nObj; o++){
                    if(ind.objectives[o] > utopia[o]) utopia[o] = ind.objectives[o];
                    if(ind.objectives[o] < nadir[o]) nadir[o] = ind.objectives[o];
                }
            }

            // Compute per-objective range for normalization
            std::vector<float> range(nObj);
            for(size_t o = 0; o < nObj; o++){
                range[o] = utopia[o] - nadir[o];
                if(range[o] < 1e-12f) range[o] = 1.0f; // avoid division by zero
            }

            // Compute normalized Euclidean distance to Utopia for each individual
            std::vector<std::pair<float, int>> distances;
            distances.reserve(individuals.size());
            for(int i = 0; i < individuals.size(); i++){

                float dist = 0.0f;
                for(size_t o = 0; o < nObj; o++){
                    float normalized = (utopia[o] - individuals[i].objectives[o]) / range[o];
                    dist += normalized * normalized;
                }
                distances.emplace_back(std::sqrt(dist), i);
            }

            // Sort by distance (ascending = closest to Utopia first)
            std::sort(distances.begin(), distances.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });

            // Select E closest, avoiding duplicates
            std::unordered_set<int> alreadySelected;
            int added = 0;
            for(const auto& [dist, idx] : distances){
                if(added >= E) break;
                if(alreadySelected.count(idx)) continue;
                indicesElite.push_back(selection.size());
                selection.push_back(individuals[idx]);
                individuals[idx].nBest++;
                alreadySelected.insert(idx);
                if(individuals[idx].fitness > bestFit) bestFit = individuals[idx].fitness;
                added++;
            }
        }

        /**
         * @brief Applies gamma mutation to all non-elite JE nodes in the population.
         * @param probability Probability that each node's gamma is resampled.
         * @param justUsedNodes If true, only mutates nodes that were used during traversal.
         */
        void callGammaMutation(float probability, bool justUsedNodes = false) {
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end()) continue;
                bool changed = false;
                for (auto& node : individuals[i].innerNodes) {
                    if (node.type != "JE") continue;
                    if (justUsedNodes && !node.used) continue;
                    if (node.gammaMutation(probability)) changed = true;
                }
                if (changed) individuals[i].structureChangedThisGen = true;
            }
        }

        /**
         * @brief Applies alpha mutation to all non-elite JE nodes in the population.
         * @param probability Probability that each node's alpha is resampled.
         * @param justUsedNodes If true, only mutates nodes that were used during traversal.
         */
        void callAlphaMutation(float probability, bool justUsedNodes = false) {
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                if (std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end()) continue;
                bool changed = false;
                for (auto& node : individuals[i].innerNodes) {
                    if (node.type != "JE") continue;
                    if (justUsedNodes && !node.used) continue;
                    if (node.alphaMutation(probability)) changed = true;
                }
                if (changed) individuals[i].structureChangedThisGen = true;
            }
        }



};

#endif
