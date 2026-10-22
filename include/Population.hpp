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
#include <limits>
#include <memory>
#include <cstdint>
#include "Network.hpp"

/**
 * @class Population 
 * @brief Manages the evolutionary population of GNP individuals (networks).
 *
 * @details
 * The Population class implements the evolutionary framework for Genetic Network Programming (GNP).
 * It provides all necessary operations for evolutionary optimization:
 * 
 * - **Population Management**: Initialization and maintenance of multiple network individuals
 * - **Fitness Evaluation**: Applying fitness functions (classification accuracy, Gymnasium environments via
 *   the Python bindings, or any user-defined function writing fitness/caseScores)
 * - **Selection**: Tournament selection with elitism (to preserve best individuals)
 * - **Genetic Operators**: 
 *   - Edge mutation (topology changes)
 *   - Boundary mutation (parameter tuning)
 *   - Crossover (recombination of network structures)
 *   - Node addition/deletion (structural evolution)
 * 
 * The population evolves over generations through iterative applying fitness measurments,
 * selection, and mutation operators, which gradually improves the collective fitness and
 * discovering effective network structures for the target problem. Within a generation the steps
 * must run in a fixed order (see Step), which the population checks unless checkOrder is false.
 * A small tutorial can be found here:
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
        int lastStep_ = -1;         // highest Step since the last evaluation; -1 = never evaluated (see checkStep())
        std::string lastStepName_;  // method that reached lastStep_, for the error message

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
        std::vector<int> indicesElite; /**< Indices of elite individuals (protected from mutation) */
        int maxNetworkSize;
        std::vector<int> nFeatureValues; /**< stores the number of feature values */
        // Diagnostics of the LAST crossover() call; only type="semantic" fills them
        int crossoverPairsApplied = 0;   /**< Parent pairs that actually entered the exchange. Together with the
                                              counters below it tells "no effect" apart from "never fired". */
        int crossoverNodesMatched = 0;   /**< Node pairs the role matching found, summed over all pairs. */
        int crossoverNodesExchanged = 0; /**< Nodes actually transferred donor -> recipient. */
        int crossoverNodesSkipped = 0;   /**< Nodes drawn for exchange but skipped: an edge was untranslatable.
                                              If this dominates crossoverNodesExchanged, boundaryTolerance is too
                                              tight for a gene to travel with its whole neighbourhood (see the
                                              all-or-nothing rule in the semantic branch). */
        int nextTransplantBlockID = 1; /**< Monotonically increasing counter used to tag each transplanted
                                             sub-graph produced by crossover(type="caseSpecialist") with a unique
                                             Node::transplantBlockID (0 is reserved / unused, -1 means "no block"). */
        /** @endcond */

        /** @name Fitness statistics */
        /** @{ */
        /**
         * @brief Highest fitness among the current individuals.
         *
         * @details Computed from Network::fitness on every call, so it always describes the
         * individuals as they are now: after an evaluation the evaluated population, after a
         * selection the selected one, after a validation run the validation result. Mutation
         * and crossover do not touch fitness, so until the next evaluation the value still
         * refers to the last one. Individuals never evaluated carry the lowest float.
         *
         * @return Maximum fitness, or NaN for an empty population.
         */
        float bestFit() const {
            if(individuals.empty()) return std::numeric_limits<float>::quiet_NaN();
            float best = individuals[0].fitness;
            for(const Network& ind : individuals) best = std::max(best, ind.fitness);
            return best;
        }

        /**
         * @brief Mean fitness over all current individuals, elite included.
         * @details Computed on every call, like bestFit().
         * @return Mean fitness, or NaN for an empty population.
         */
        float meanFitness() const {
            if(individuals.empty()) return std::numeric_limits<float>::quiet_NaN();
            double sum = 0.0; // double: float loses precision over many large rewards
            for(const Network& ind : individuals) sum += ind.fitness;
            return static_cast<float>(sum / individuals.size());
        }

        /**
         * @brief Lowest fitness among the current individuals.
         * @details Computed on every call, like bestFit().
         * @return Minimum fitness, or NaN for an empty population.
         */
        float minFitness() const {
            if(individuals.empty()) return std::numeric_limits<float>::quiet_NaN();
            float worst = individuals[0].fitness;
            for(const Network& ind : individuals) worst = std::min(worst, ind.fitness);
            return worst;
        }
        /** @} */

        /** @name Self-adaptive mutation strength (Rechenberg 1/5-success-rule) */
        /** @{ */
        using SuccessCriterion = std::function<float(const Network&)>; /**< Exchangeable per-individual scoring function for updateAdaptiveK(). May be a continuous score in [0,1] (e.g. caseConsistencyRatio()) or a legacy bool-returning predicate (implicitly converted to 1.0f/0.0f, e.g. caseConsistentSuccess()). */

        float adaptiveK_ = 1.0f; /**< Current self-adapted k, used as the mutation-strength parameter (see updateAdaptiveK()). */
        std::deque<float> successRateHistory_; /**< Rolling window of per-generation (mean) scores -- smooths the signal over multiple generations (see windowSize in updateAdaptiveK()). Most recent at the back. */
        /** @} */

        /** @name Order of the steps of a generation */
        /** @{ */
        /**
         * @brief Place of each step in a generation; steps must run in this order.
         *
         * @details
         * The evaluation writes Node::used, the traversal counters, fitness, caseScores and
         * visitedNodesPerCase. Every later step reads some of them, and every step that changes
         * the networks makes some of them stale:
         * - selection compares fitness, which no longer describes a network once it was varied;
         * - callAgeUnusedNodes() and callAddDelNodes() need usage flags that still match the
         *   networks: after a mutation or crossover a node on the active path can be marked
         *   unused and would be deleted, which changes the fitness;
         * - crossover(type="caseSpecialist*") reads visitedNodesPerCase by node index, which
         *   callAddDelNodes() shifts by deleting nodes;
         * - crossover picks donors by fitness and (semantic, matchOnlyUsed) nodes by usage,
         *   both stale after a mutation;
         * - updateAdaptiveK() sets k for the mutations of this generation, and
         *   capLineageAfterMutation() needs every change of the generation.
         *
         * A step may only run if no step with a higher place ran since the last evaluation;
         * every step is optional. AgeUnusedNodes, Selection and CaseSpecialist run at most once
         * per evaluation. checkStep() enforces this, `checkOrder = false` switches it off.
         */
        enum class Step : int {
            Evaluation = 0,     ///< gymnasium(), callTraversePath(), accuracy(), applyFitness(), markEvaluated(); calculateParetoObjectives() shares the place
            AgeUnusedNodes = 1, ///< callAgeUnusedNodes()
            Selection = 2,      ///< tournamentSelection(), lexicaseSelection(), paretoTournamentSelection()
            AdaptiveK = 3,      ///< updateAdaptiveK()
            CaseSpecialist = 4, ///< crossover(type="caseSpecialist", "caseSpecialistAppend", "caseSpecialistReplace")
            AddDelNodes = 5,    ///< callAddDelNodes()
            Crossover = 6,      ///< crossover(), all other types
            Mutation = 7,       ///< callEdgeMutation(), callBoundaryMutation*(), callGammaMutation(), callAlphaMutation()
            CapLineage = 8      ///< capLineageAfterMutation()
        };

        bool checkOrder = true; /**< If true (default), a step out of the order of Step throws std::logic_error.
                                     Set to false to call single operators on purpose, e.g. in tests. */

        /**
         * @brief Marks the population as freshly evaluated; the next generation's steps start over.
         *
         * @details Every evaluation of this class calls it. An evaluation outside of it -- e.g.
         * fracnetics.parallel_eval.ParallelEvaluator or a fitness computed entirely in Python --
         * must call it afterwards, otherwise the steps of the new generation are checked against
         * those of the previous one.
         */
        void markEvaluated(){
            lastStep_ = static_cast<int>(Step::Evaluation);
            lastStepName_ = "the evaluation";
        }

        /**
         * @brief Checks that a step runs in the order of Step and records it.
         *
         * @details Throws if a step with a higher place ran since the last evaluation, or if
         * callAgeUnusedNodes(), a selection or a caseSpecialist crossover runs a second time. The
         * check starts with the first evaluation: before it, nothing is recorded, so freshly
         * created or unpickled populations accept any step. The step is recorded even with
         * `checkOrder == false`, so switching the check back on mid-generation stays consistent.
         *
         * @param step Place of the step.
         * @param name Name of the calling method, used in the error message.
         * @throws std::logic_error if the step is out of order and checkOrder is true.
         */
        void checkStep(Step step, const std::string& name){
            if(lastStep_ < 0) return; // never evaluated: nothing to check against
            const int place = static_cast<int>(step);
            const bool once = step == Step::AgeUnusedNodes || step == Step::Selection
                              || step == Step::CaseSpecialist;
            const bool twice = once && place == lastStep_;
            if(checkOrder && (place < lastStep_ || twice)){
                const std::string what = twice
                    ? name + " ran twice since the last evaluation."
                    : name + " after " + lastStepName_ + " without an evaluation in between.";
                throw std::logic_error(
                    "Population: " + what + " The steps of a generation must run in this order "
                    "(each optional): evaluation, callAgeUnusedNodes (once), selection (once), "
                    "updateAdaptiveK, crossover caseSpecialist* (once), callAddDelNodes, crossover, "
                    "mutations, capLineageAfterMutation. Later steps leave the usage flags and the "
                    "fitness of the last evaluation stale (see Population::Step). Evaluate again "
                    "first, or set checkOrder = False to call single operators on purpose.");
            }
            if(place >= lastStep_){
                lastStep_ = place;
                lastStepName_ = name;
            }
        }
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
         * @note Counts as the evaluation of the generation (see Step, markEvaluated()).
         */
        void callTraversePath(
                const std::vector<std::vector<float>>& X,
                int dMax
                ){
            for (auto& network : individuals){
                network.traversePath(X,dMax);
            }
            markEvaluated();
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
         * @note Counts as the evaluation of the generation (see Step, markEvaluated()).
         */
        template <typename FuncFitness>
        void applyFitness(FuncFitness&& func){
            for (auto& network : individuals){
                func(network);
            }
            markEvaluated();
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
         * Afterwards bestFit(), meanFitness() and minFitness() describe the selected population.
         *
         * Tournament selection provides tunable selective pressure: larger N increases pressure
         * (stronger individuals more likely to be selected), while smaller N maintains diversity.
         * Elitism ensures best solutions are never lost, providing monotonic improvement guarantee.
         * 
         * @param N Tournament size (number of individuals per tournament)
         * @param E Elite size (number of best individuals to preserve unchanged)
         * @param useLineageFitness If true, the selection criterion is the lineage estimate
         *        (`lineageMean` over ALL cases this lineage has seen, see lineageZ) instead of
         *        the raw current fitness -- makes selection robust against generalisation
         *        outliers (e.g. a single very hard case in the current generation). Applies
         *        ONLY to individuals with `lineageN >= minLineageN` (bootstrap protection); all
         *        others (e.g. young lineages right after initialisation) fall back to the raw
         *        `fitness` until enough observations exist. The fitness statistics
         *        (bestFit() etc.) always read the raw fitness, for consistent logging/plotting.
         * @param minLineageN Minimum number of lineage observations from which `lineageMean` is
         *        used instead of the raw fitness (only relevant if `useLineageFitness == true`).
         *        Default 10.
         * @param lineageZ z of the lower confidence bound Network::lineageLCB() used as criterion.
         *        The LCB penalises lineages with small lineageN/high uncertainty, so "lucky
         *        newcomers" that just crossed minLineageN cannot displace established, more
         *        precisely estimated lineages through sampling noise (winner's curse). 0.0
         *        (default) reduces the LCB to the plain lineageMean (previous behaviour).
         *
         * @note N ≥ 2 for meaningful selection pressure
         * @note Population size remains constant at ni
         * 
         */
        void tournamentSelection(int N, int E, bool useLineageFitness = false, int minLineageN = 10, float lineageZ = 0.0f){
            checkStep(Step::Selection, "tournamentSelection()");
            // Raw fitness, or the lineage LCB once enough observations exist (see docstring)
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
            }
            setElite(E, individuals, selection, useLineageFitness, minLineageN, lineageZ);
            individuals = std::move(selection);
            // set frozenExperience flag back for non-elite
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
        }

        /**
         * @brief Performs Epsilon-Lexicase Selection over test cases: the cases of an evaluation
         *        (episodes, samples, ...), user-defined objectives, or both combined.
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
         * All types maximise (higher = better). The data they read is filled by the evaluation:
         * Network::caseScores by every evaluator, Network::objectivesPerCase and
         * Network::lexicaseObjectives only if objectives were provided (e.g. the episodeObjectives
         * hook of the Gymnasium evaluator).
         *
         * **type = "cases"**: one test case per case (network.caseScores[caseIdx]). For each case
         * in the shuffled order the score is maximised directly, with the MAD- or fixed-epsilon
         * tolerance.
         *
         * **type = "casesStandardized"**: same test cases as "cases", but the scores of each case
         * are z-transformed over the whole population once per generation, and the tolerance is a
         * fixed number of standard deviations (`epsilon`, negative defaults to 0.5) instead of the
         * MAD automatic. This makes the selection pressure independent of the shape and the scale
         * of the scores and removes the difficulty of a case, while -- unlike a rank transform --
         * keeping the magnitudes that epsilon is meant to judge.
         * Why not the raw MAD of "cases": with bimodal scores (e.g. success vs. failure) the MAD
         * describes the spread of whichever group holds the median, not the resolution among
         * near-equal candidates, and near an even split it jumps erratically, up to almost the gap
         * between the groups -- the test case then only separates the two groups. The standard
         * deviation changes smoothly with the split, so a small fixed fraction of it keeps
         * resolving within a group.
         * Standardising over the population rather than over the shrinking candidate pool is
         * deliberate (the "semi-dynamic" variant), so a fixed epsilon does not get more
         * permissive along the filter chain. Cases on which all individuals are identical carry
         * no information and filter nobody.
         *
         * **type = "casesRank"**: same test cases as "cases", but each filter cuts by RANK
         * instead of by score: the remaining candidates are sorted by their score on that case
         * and the best (epsilon + 1) ranks are kept, ties surviving together. The ranks are
         * recomputed over the remaining pool at every step, so the selection pressure stays
         * constant along the chain. `epsilon` is therefore a number of rank positions, not a
         * score margin, and the MAD automatic is not available (the MAD of ranks is ~n/4 whatever
         * the data); a negative epsilon defaults to 3. On ranks the sharpness is independent of
         * the shape and scale of the scores, and the case difficulty drops out.
         * WARNING: an absolute rank cut collapses the pool to epsilon+1 candidates in a
         * single step, after which every remaining test case filters nobody -- only the
         * first one or two cases ever decide a parent. Prefer "casesStandardized".
         *
         * **type = "objectives" (default)**: test cases are the objectives of
         * Network::lexicaseObjectives, i.e. each objective averaged over the cases. Each objective
         * is filtered with its own tolerance, so objectives on different scales need no
         * normalisation.
         *
         * **type = "objectivesPerCase"**: test cases are the (case x objective) pairs
         * (network.objectivesPerCase[caseIdx][objIdx]), giving K*N test cases for K objectives
         * and N cases instead of K or N. With one scalar per case, individuals on a converged
         * plateau become indistinguishable (their scores lie closer together than the case-to-case
         * noise), while separate objectives still measure different things and separate them. The
         * whole grid is shuffled as one, so neither cases nor objectives are systematically
         * decided first.
         *
         * **Algorithm per selection event** (all types):
         * 1. Start with the full population as candidate pool.
         * 2. Draw a random permutation of the test cases.
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
         *                instead. "casesStandardized" reads it in standard deviations,
         *                "casesRank" in rank positions (see above).
         * @param type One of "objectives" (default), "objectivesPerCase", "cases",
         *             "casesStandardized" or "casesRank" -- selects which test case
         *             decomposition to use (see above). Throws std::invalid_argument
         *             for any other value.
         *
         * @pre For "objectives"/"objectivesPerCase": all individuals carry equally sized,
         *      non-empty lexicaseObjectives/objectivesPerCase.
         * @pre For the "cases*" types: all individuals carry equally sized, non-empty caseScores.
         *
         * @note Population size remains constant at ni.
         * @note Elite handling mirrors tournamentSelection() for compatibility with the rest
         *       of the pipeline.
         */
        void lexicaseSelection(int E, float epsilon = -1.0f, std::string type = "objectives"){
            if(type != "objectives" && type != "cases" && type != "objectivesPerCase"
               && type != "casesRank" && type != "casesStandardized"){
                throw std::invalid_argument("lexicaseSelection: unknown type \"" + type + "\" (expected \"objectives\", \"objectivesPerCase\", \"cases\", \"casesStandardized\" or \"casesRank\")");
            }

            // MAD-based tolerance (or the fixed epsilon) for one set of values
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

            maxNetworkSize = individuals[0].innerNodes.size();

            std::vector<int> allIndices(individuals.size());
            std::iota(allIndices.begin(), allIndices.end(), 0);

            const size_t nObjectives = individuals[0].lexicaseObjectives.size(); // used if type=="objectives"
            const size_t nCases = individuals[0].caseScores.size();          // used if type=="cases"
            // Empty test cases would silently degrade the selection to a random draw
            if(type == "objectives" && nObjectives == 0){
                throw std::invalid_argument(
                    "lexicaseSelection(type=\"objectives\"): Network::lexicaseObjectives is empty -- "
                    "evaluate with objectives first (e.g. the episodeObjectives hook)");
            }
            if(type.rfind("cases", 0) == 0 && nCases == 0){
                throw std::invalid_argument(
                    "lexicaseSelection(type=\"" + type + "\"): Network::caseScores is empty -- evaluate first");
            }

            std::vector<int> objectiveOrder(nObjectives);
            std::iota(objectiveOrder.begin(), objectiveOrder.end(), 0);

            std::vector<int> caseOrder(nCases);
            std::iota(caseOrder.begin(), caseOrder.end(), 0);

            // type=="casesRank": epsilon in RANK positions; no MAD automatic (MAD of ranks is ~n/4)
            const int rankTolerance = (epsilon < 0.0f) ? 3 : static_cast<int>(std::lround(epsilon));

            // type=="casesStandardized": z-transform each case over the WHOLE population (see docstring)
            std::vector<std::vector<float>> standardizedByCase;
            if(type == "casesStandardized"){
                standardizedByCase.assign(nCases, std::vector<float>(individuals.size(), 0.0f));
                for(size_t caseIdx = 0; caseIdx < nCases; caseIdx++){
                    double sum = 0.0;
                    for(const auto& ind : individuals) sum += ind.caseScores[caseIdx];
                    double mean = sum / static_cast<double>(individuals.size());
                    double sqSum = 0.0;
                    for(const auto& ind : individuals){
                        double d = ind.caseScores[caseIdx] - mean;
                        sqSum += d * d;
                    }
                    double sd = std::sqrt(sqSum / static_cast<double>(individuals.size()));
                    // All identical on this case: leave 0 so it filters nobody
                    if(sd < 1e-9){
                        continue;
                    }
                    for(size_t c = 0; c < individuals.size(); c++){
                        standardizedByCase[caseIdx][c] =
                            static_cast<float>((individuals[c].caseScores[caseIdx] - mean) / sd);
                    }
                }
            }
            // epsilon in standard deviations; negative selects a sensible default.
            const float sigmaTolerance = (epsilon < 0.0f) ? 0.5f : epsilon;

            // type=="objectivesPerCase": the test cases are the (case, objective) pairs,
            // flattened as t = caseIdx*nObjPerCase + objIdx.
            const size_t nObjPerCase = individuals[0].objectivesPerCase.empty()
                ? 0 : individuals[0].objectivesPerCase[0].size();
            std::vector<int> gridOrder;
            if(type == "objectivesPerCase"){
                if(individuals[0].objectivesPerCase.empty()){
                    throw std::invalid_argument(
                        "lexicaseSelection(type=\"objectivesPerCase\"): Network::objectivesPerCase is "
                        "empty -- evaluate with objectives first (e.g. the episodeObjectives hook)");
                }
                gridOrder.resize(individuals[0].objectivesPerCase.size() * nObjPerCase);
                std::iota(gridOrder.begin(), gridOrder.end(), 0);
            }
            // After the argument checks, so a rejected call does not count as the selection
            checkStep(Step::Selection, "lexicaseSelection()");

            // Keep everyone within the tolerance of the best value; computed per test
            // case, so the different scales of the objectives need no normalisation
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
                } else if(type == "objectivesPerCase"){
                    // Shuffle the whole (case x objective) grid as one, so neither is decided first
                    std::shuffle(gridOrder.begin(), gridOrder.end(), *generator);

                    for(int t : gridOrder){
                        if(candidates.size() <= 1) break;
                        const size_t caseIdx = static_cast<size_t>(t) / nObjPerCase;
                        const size_t objIdx  = static_cast<size_t>(t) % nObjPerCase;
                        filterOnTestCase(candidates, [&](int c){
                            return individuals[c].objectivesPerCase[caseIdx][objIdx];
                        });
                    }
                } else if(type == "casesStandardized"){
                    // Like "cases", on z-scores with a fixed tolerance in standard deviations
                    std::shuffle(caseOrder.begin(), caseOrder.end(), *generator);

                    for(int caseIdx : caseOrder){
                        if(candidates.size() <= 1) break;
                        const std::vector<float>& column = standardizedByCase[caseIdx];
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
                } else if(type == "casesRank"){
                    // ─── Rank-based per case, ranks recomputed on the remaining pool (see docstring) ───
                    std::shuffle(caseOrder.begin(), caseOrder.end(), *generator);

                    for(int caseIdx : caseOrder){
                        if(candidates.size() <= 1) break;

                        // Cut at rank rankTolerance + 1; ties survive together
                        std::vector<float> values;
                        values.reserve(candidates.size());
                        for(int c : candidates){
                            values.push_back(individuals[c].caseScores[caseIdx]);
                        }
                        std::sort(values.begin(), values.end(), std::greater<float>());
                        size_t cutIndex = std::min(static_cast<size_t>(std::max(0, rankTolerance)),
                                                   values.size() - 1);
                        float cutoff = values[cutIndex];

                        std::vector<int> filtered;
                        filtered.reserve(candidates.size());
                        for(int c : candidates){
                            if(individuals[c].caseScores[caseIdx] >= cutoff){
                                filtered.push_back(c);
                            }
                        }
                        candidates = std::move(filtered);
                    }
                } else { // type == "cases"
                    // Draw a fresh, independent case evaluation order for this selection event
                    std::shuffle(caseOrder.begin(), caseOrder.end(), *generator);

                    for(int caseIdx : caseOrder){
                        if(candidates.size() <= 1) break;
                        // Single criterion per case: maximise the score.
                        filterOnTestCase(candidates, [&](int c){
                            return individuals[c].caseScores[caseIdx];
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
            }

            setElite(E, individuals, selection);
            individuals = std::move(selection);
            // set frozenExperience flag back for non-elite
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
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
         * @param useLineageFitness If true, the elite is ranked by `lineageMean` instead of the raw
         *        `fitness` -- but only for individuals with `lineageN >= minLineageN` (others fall
         *        back to their raw fitness). Default false = raw fitness only, so other callers
         *        (e.g. lexicaseSelection()) are unaffected.
         * @param minLineageN Minimum number of lineage observations from which lineageMean is used
         *        for the elite ranking (only relevant if useLineageFitness=true).
         * @param lineageZ Confidence multiplier for the lower confidence bound (LCB) of the lineage
         *        mean (see Network::lineageLCB()). 0.0 (default) = plain lineageMean without an
         *        uncertainty discount. Larger values penalise lineages with small lineageN/high
         *        variance more, so "lucky newcomers" that just crossed minLineageN cannot displace
         *        established champions through sampling noise (winner's curse).
         *
         * @note Elite indices are used to protect elite from mutation operations
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
            }
        }

        /**
         * @brief Caps the lineage observation count (lineageN/lineageSuccessN) ONLY for
         *        individuals that were ACTUALLY changed by mutation or crossover this generation
         *        (individuals[i].structureChangedThisGen == true) -- the mutation counterpart of
         *        priorCap in crossover(). Call once per generation, AFTER all mutation operators
         *        of the generation have run (however many were used -- the operation is monotone
         *        and therefore safe to call repeatedly).
         *
         * @details Without this cap, conditioned on an actual change, either (a) a pure mutation
         * lineage would accumulate its `lineageMean` without bound and grow very sluggish towards
         * new observations after many generations, OR (b) -- if the cap were applied blanket to
         * ALL non-elite individuals regardless of a real structural change (an earlier bug) --
         * unchanged individuals would needlessly lose history every generation, letting the
         * current (noisy) batch of cases dominate lineageMean more than intended. Unchanged lineages
         * therefore keep their full accumulated history.
         *
         * The flag `structureChangedThisGen` is reset here for ALL individuals after reading it
         * (whether capped or not), so the next generation starts at false again.
         *
         * @param priorCap Upper bound for lineageN/lineageSuccessN. Default 5 (as in crossover()).
         */
        void capLineageAfterMutation(int priorCap = 5) {
            checkStep(Step::CapLineage, "capLineageAfterMutation()");
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                if (!isElite && individuals[i].structureChangedThisGen) {
                    individuals[i].capLineageStats(priorCap);
                }
                individuals[i].structureChangedThisGen = false; // reset for the next generation
            }
        }

        /**
         * @brief Lineage-based success criterion for updateAdaptiveK().
         *
         * @details
         * Instead of comparing an individual's fitness of THIS generation against a single,
         * equally noisy prior generation -- which would re-inject the case-sampling noise this
         * mechanism is meant to filter out -- this criterion compares against the lineage's own
         * long-run, uncertainty-aware baseline (Network::lineageLCB()), aggregated over ALL cases
         * the lineage has ever been evaluated on.
         *
         * "Success" = fitness this generation is at or above that baseline, i.e. the lineage
         * performs consistently with -- or better than -- its own track record.
         *
         * @param ind Individual to evaluate.
         * @param z   Confidence multiplier passed to Network::lineageLCB() (larger z = more
         *            conservative baseline). Default 1.0.
         * @return true if lineageN <= 1 (not enough history yet -- neutral/optimistic default),
         *         or if ind.fitness >= ind.lineageLCB(z).
         * @note Requires lineage tracking during evaluation (trackLineage=True).
         * @see Network::lineageLCB()
         */
        static bool lineageRobustSuccess(const Network& ind, float z = 1.0f) {
            if (ind.lineageN <= 1) return true; // not enough history yet -> neutral, do not count as failure
            return ind.fitness >= ind.lineageLCB(z);
        }

        /**
         * @brief Case-consistent, lineage-based success criterion for updateAdaptiveK().
         *
         * @details
         * Stronger than lineageRobustSuccess(): instead of judging success on the AVERAGE over
         * this generation's cases (which a single very good or very bad case can dominate), it
         * requires the improvement to be CONSISTENT across the cases -- at least
         * `requiredFraction` of this generation's case scores (Network::caseScores) must
         * individually reach or exceed the lineage's long-run baseline (Network::lineageLCB()).
         *
         * This targets sampling luck: a mutation that only looks good because it drew a few
         * easy cases this generation (while still failing hard ones) does not count as a success.
         *
         * @param ind               Individual to evaluate.
         * @param requiredFraction  Minimum fraction of case scores that must be >= lineageLCB(z).
         *                          1.0 = ALL cases, lower values (e.g. 0.8) relax this. Default 1.0.
         * @param z                 Confidence multiplier passed to Network::lineageLCB(). Default 1.0.
         * @return true if lineageN <= 1 or caseScores is empty (neutral default), otherwise true
         *         iff the fraction of case scores >= lineageLCB(z) is >= requiredFraction.
         * @note Requires lineage tracking during evaluation (trackLineage=True).
         * @see lineageRobustSuccess()
         */
        static bool caseConsistentSuccess(const Network& ind, float requiredFraction = 1.0f, float z = 1.0f) {
            if (ind.lineageN <= 1 || ind.caseScores.empty()) return true; // not enough history/data yet -> neutral

            float baseline = ind.lineageLCB(z);
            int nAtOrAbove = 0;
            for (float v : ind.caseScores) {
                if (v >= baseline) nAtOrAbove++;
            }
            float fraction = static_cast<float>(nAtOrAbove) / static_cast<float>(ind.caseScores.size());
            return fraction >= requiredFraction;
        }

        /**
         * @brief Absolute-threshold success criterion for updateAdaptiveK(), without lineage data.
         *
         * @details
         * Looks only at THIS generation's case scores (Network::caseScores), so it also works
         * when lineage tracking is off -- where lineageN stays 0 and the lineage criteria always
         * return their neutral "true", saturating the success rate and driving adaptiveK_ to
         * kMin/kMax immediately.
         *
         * Success requires at least `requiredFraction` of the case scores to reach `threshold`,
         * i.e. "(almost) all cases this generation were solved".
         *
         * @param ind               Individual to evaluate.
         * @param threshold         Score a case must reach to count as solved (problem-specific).
         * @param requiredFraction  Minimum fraction of solved cases. 1.0 = ALL cases. Default 1.0.
         * @return true if caseScores is empty (neutral default), otherwise true iff the fraction
         *         of case scores >= threshold is >= requiredFraction.
         * @see caseConsistencyRatio()
         */
        static bool caseThresholdSuccess(const Network& ind, float threshold, float requiredFraction = 1.0f) {
            if (ind.caseScores.empty()) return true; // no data yet -> neutral

            int nAtOrAbove = 0;
            for (float v : ind.caseScores) {
                if (v >= threshold) nAtOrAbove++;
            }
            float fraction = static_cast<float>(nAtOrAbove) / static_cast<float>(ind.caseScores.size());
            return fraction >= requiredFraction;
        }

        /**
         * @brief Continuous, scale-independent case-consistency score for updateAdaptiveK() (bundled DEFAULT).
         *
         * @details
         * Unlike the binary criteria, and unlike caseThresholdSuccess(), whose absolute threshold
         * is often only reachable late in a run, this returns a CONTINUOUS score in [0,1] from the
         * RELATIVE spread of this generation's case scores (Network::caseScores) -- meaningful
         * from generation 1 onward, at any absolute fitness level:
         *
         *     consistency = max(0, min(caseScores)) / max(caseScores)
         *
         * 1.0 = the worst case did just as well as the best one. Values near 0 mean a severe
         * outlier case relative to the best one (fragile structure). updateAdaptiveK() averages it
         * over the non-elite population and smooths it over `windowSize` generations.
         *
         * @param ind Individual to evaluate.
         * @return 1.0 if fewer than 2 case scores or max(caseScores) <= 0 (not enough data /
         *         degenerate case -- neutral default), otherwise
         *         max(0, min(caseScores)) / max(caseScores), clamped to [0,1].
         * @note Assumes scores where larger is better and 0 is a natural floor.
         * @see caseThresholdSuccess()
         * @see caseConsistentSuccess()
         */
        static float caseConsistencyRatio(const Network& ind) {
            if (ind.caseScores.size() < 2) return 1.0f; // not enough cases this gen -> neutral

            float minV = *std::min_element(ind.caseScores.begin(), ind.caseScores.end());
            float maxV = *std::max_element(ind.caseScores.begin(), ind.caseScores.end());
            if (maxV <= 0.0f) return 1.0f; // degenerate (all non-positive) -> neutral, avoid div-by-zero/negative ratio

            minV = std::max(minV, 0.0f); // clamp a negative worst case to 0 so the ratio stays in [0,1]
            return std::min(1.0f, minV / maxV);
        }


        /**
         * @brief Self-adaptive mutation-strength k, following Rechenberg's 1/5-success-rule,
         *        with a fully exchangeable success criterion.
         *
         * @details
         * Must be called once per generation, AFTER this generation's evaluation/selection
         * (evaluation and lexicaseSelection() etc. -- so that individuals[i].fitness
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
         *      is already robust against case noise, e.g. caseConsistencyRatio()): score > target
         *      -> already consistently succeeding -> `decFactor` (shrink k, preserve/stabilize
         *      the working structure); score < target -> not consistently succeeding
         *      -> `incFactor` (grow k, explore out of the insufficient structure).
         * 4. If `useRechenberg == false`: history is still recorded (for diagnostics/plots),
         *    but adaptiveK_ is left untouched -- use setAdaptiveK() to set a fixed value
         *    manually in that case.
         *
         * The scoring function is fully exchangeable: pass any `(const Network&) -> float`
         * (or legacy `-> bool`) callable. The bundled default (caseConsistencyRatio()) is a
         * CONTINUOUS, scale-independent measure of this generation's per-case spread
         * (worst/best case ratio) -- meaningful from generation 1 onward, unlike the
         * threshold-based alternative (caseThresholdSuccess()), which stays stuck at one
         * extreme for as long as the absolute threshold is out of reach. The lineage-based
         * alternatives (caseConsistentSuccess()/lineageRobustSuccess()) require lineage
         * tracking during evaluation (trackLineage=True).
         *
         * @param successCriterion  Per-individual scoring function, [0,1] (or bool). Default: caseConsistencyRatio().
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
         * @see caseConsistencyRatio()
         * @see caseThresholdSuccess()
         * @see caseConsistentSuccess()
         * @see adaptiveK_
         */
        void updateAdaptiveK(
                SuccessCriterion successCriterion = [](const Network& ind){ return caseConsistencyRatio(ind); },
                bool useRechenberg = true,
                bool invert = false,
                int windowSize = 10,
                float targetSuccessRate = 0.2f,
                float incFactor = 1.22f,
                float decFactor = 0.82f,
                float kMin = 0.1f,
                float kMax = 5.0f)
        {
            checkStep(Step::AdaptiveK, "updateAdaptiveK()");
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
         * mutation to preserve the best solutions found so far. Frozen nodes are skipped as well.
         *
         * @see Node::edgeMutation()
         *
         * @param probInnerNodes Probability (in [0.0, 1.0]) that each edge of inner nodes will be mutated
         * @param probStartNode Probability (in [0.0, 1.0]) that the start node's edge will be mutated
         * @param justUsedNodes If true, only applies edge mutation to nodes that were used during traversal (node.used == true).
         * If false, applies to all nodes regardless of usage.
         * @param k If > 0, replaces both probabilities with min(1, k/N), N being the network's number of
         * (used) edges, i.e. about k mutated edges per network (see Node::edgeMutation(), adaptiveK_).
         *
         * @warning tournamentSelection() must have been called to set indicesElite
         *
         */
        void callEdgeMutation(float probInnerNodes, float probStartNode, bool justUsedNodes = false, float k = 0.0f){
            checkStep(Step::Mutation, "callEdgeMutation()");
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
         * @param k If > 0, each boundary is mutated with min(1, k/N). N is the number of EDGES of
         *  the eligible nodes (countEdges()), not of boundaries: a judgment node has one boundary
         *  fewer than edges and processing nodes count too, so k = 1 moves fewer than one boundary
         *  per network (0.74 measured in a LunarLander experiment).
         * @param N Overwritten when k > 0.
         * @param stepName Name of the calling method for the order check (see checkStep()).
         *
         * @note This is an internal template used by specialized boundary mutation methods
         */
        template <typename FuncMutation>
        void applyBoundaryMutation(FuncMutation&& func, bool justUsedNodes = false, float k = 0.0f, int N = 1,
                                   const std::string& stepName = "applyBoundaryMutation()") {
            checkStep(Step::Mutation, stepName);
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
            }, justUsedNodes, k, 1, "callBoundaryMutationUniform()");
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
         * @param sigma Relative standard deviation, multiplied by the smaller gap to the neighbouring
         *  boundaries (see Node::boundaryMutationNormal()).
         * @param justUsedNodes If true, only applies mutation to judgment nodes that were used during traversal (node.used == true).
         * @param k If > 0, replaces probability with min(1, k/N); see applyBoundaryMutation() for N.
         * 
         * @note Smaller sigma → more conservative, larger sigma → more exploratory
         */
        void callBoundaryMutationNormal(const float probability, const float sigma, bool justUsedNodes, float k = 0.0f){
            applyBoundaryMutation([=](Node& node, const additionalMutationParam&, bool justUsedNodes, float k, int N){
                return node.boundaryMutationNormal(probability, sigma, k, N);
            }, justUsedNodes, k, 1, "callBoundaryMutationNormal()");
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
            }, justUsedNodes, k, 1, "callBoundaryMutationNetworkSizeDependingSigma()");
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
            }, justUsedNodes, k, 1, "callBoundaryMutationEdgeSizeDependingSigma()");
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
            }, justUsedNodes, k, 1, "callBoundaryMutationFractal()");
        }

        /**
         * @brief Performs crossover (recombination) between pairs of individuals in the population.
         * 
         * @details
         * This method implements crossover, a genetic operator that exchanges
         * parts of the gene structure between parent networks to create offspring. The process:
         * 
         * **Pairing** (all types except "caseSpecialist*"):
         * 1. Shuffles all individual indices randomly
         * 2. Pairs adjacent individuals in the shuffled order (0-1, 2-3, 4-5, etc.)
         * 3. Skips pairs of two elites; if one parent is elite, it only donates and stays unchanged
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
         * **Lineage statistics** are blended in the direction of the actual gene flow, once per pair:
         * one-sided (Network::blendLineageWith()) when an elite or the fitter parent donates, symmetric
         * otherwise -- both parents then read the original values, so the call order cannot matter.
         * 
         * @param propability Probability (in [0.0, 1.0]), interpreted per crossover type. Default is 1
         *  (i.e. crossover is always applied). Values outside [0.0, 1.0] throw std::invalid_argument,
         *  since std::bernoulli_distribution would otherwise be undefined behaviour.
         *  - "uniform": per NODE POSITION -- each of the min(size1, size2) positions is exchanged with
         *    this probability, so it controls HOW MUCH of the genotype is exchanged per pair.
         *  - "onepoint" / "randomWidth": per PAIR -- one draw decides whether the pair is recombined
         *    at all; if it fails the pair is left completely untouched. What is exchanged afterwards
         *    (cutpoint, subnetwork depth) stays uniformly random.
         *  - "semantic": TWO rates. This value gates the PAIR (one draw decides whether the pair
         *    recombines at all); `nodeExchangeRate` then decides how many of the role-matched
         *    nodes inside that pair are transferred. This is deliberate: a pair recombines noticeably
         *    or not at all, instead of spreading thinly over the population. Passing a negative
         *    nodeExchangeRate couples the two, i.e. the effective rate per matched node becomes
         *    propability^2 (0.05 -> 0.0025), which reproduces the earlier behaviour bit for bit.
         *  - "caseSpecialist" / "caseSpecialistAppend" / "caseSpecialistReplace": per INDIVIDUAL --
         *    one draw per host decides whether that host receives a transplant this generation.
         *    Drawn before the donor search, so a low value also skips the O(populationSize) search
         *    for that host.
         * @param type Type of the crossover:
         *  - "uniform": selects each node and exchanges them with given probability
         *  - "onepoint": draws a random cutpoint from the genotype and exchanges all nodes until this point
         *  - "randomWidth": exchanges subnetworks of different widths where all succesor nodes of a randomly selected node are exchanged
         *  - "semantic": role-based crossover -- homology is read off the node content instead of the
         *    array position. Position-based types are weakly founded for GNP: networks start small and
         *    grow/shrink via addDelNodes(), every deletion shifts all later indices, so after a few hundred
         *    generations the same position in two individuals means nothing in common. Origin markers
         *    (NEAT-style innovation IDs) would stop NEW competing conventions but cannot recognise
         *    CONVERGENT solutions: two lineages that evolve the same decision rule independently carry
         *    different IDs. A GNP node carries its role explicitly: (type, f, edges.size()) is the coarse
         *    role, boundaries the fine tuning within it. The arity must be part of the role, otherwise an
         *    exchange would change the recipient's edge count and the threshold vectors would not be
         *    comparable.
         *    Steps: (1) group both parents by role; (2) within a group, match greedily by boundary
         *    distance -- the mean deviation of the INNER boundaries, normalised by the feature range so
         *    wide features do not weigh more (P nodes: distance 0); pairs above boundaryTolerance are not
         *    the same gene. Greedy instead of optimal (Hungarian) matching, since the groups are small and
         *    both almost always agree; (3) transfer each matched node with `nodeExchangeRate`; (4)
         *    translate its edges through the matching table. The elite only donates; otherwise the fitter
         *    parent (mean over caseScores, else fitness) donates and stays unchanged.
         *    The exchange is ALL-OR-NOTHING per node: boundaries and edges of a judgment node belong
         *    together (edge k = "feature in interval k, go there"), so mixing donor intervals with
         *    recipient targets would create a node NO parent has ever tested. A node is therefore only
         *    taken over if EVERY edge translates. Since the table only holds pairs within the tolerance, a
         *    gene travels only if it AND all its direct successors have a close counterpart -- always a
         *    complete sub-structure proven in the donor, which makes boundaryTolerance the real control.
         *    Unmatched nodes stay where they are; nothing is appended, so this type does not grow the
         *    network. Diagnostics: crossoverPairsApplied, crossoverNodesMatched/Exchanged/Skipped.
         *  - "caseSpecialist" / "caseSpecialistAppend" (identical, the short name is kept for
         *    backward compatibility): for EVERY individual (host, including elite) independently searches the whole
         *    population for the single donor individual maximizing the summed positive per-case fitness
         *    difference (donor.caseScores[s] - host.caseScores[s], summed over all s where positive).
         *    If such a donor exists, the union of nodes the donor actually traversed (Network::visitedNodesPerCase,
         *    filled by the evaluation) across those deficit cases is copied (never removed from the donor) and
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
         *    frozen). Edges leaving the copied sub-graph are redirected to random valid host nodes. This type
         *    has its OWN partner-search/insertion logic and completely bypasses the shuffle-based pairing.
         *  - "caseSpecialistReplace": same donor search and same donor-side sub-graph as
         *    "caseSpecialistAppend", but the sub-graph does not become a dormant
         *    appendix -- it OVERWRITES the host's own deficit-exclusive sub-graph, i.e. the nodes
         *    the host traverses ONLY on the deficit cases and not on the cases where the host
         *    itself is the better one. The shared backbone is therefore preserved, and with it the
         *    host's strength on its good cases; only the case-specific structure that fails is
         *    swapped out. If that set is empty (the host uses the same nodes everywhere), the host
         *    is left unchanged. Because this variant is destructive for the host, ELITE hosts are
         *    skipped (unlike the additive variant). The donor is never modified -- it only hands
         *    out copies, so one donor can serve several hosts in the same generation. Sub-graphs of
         *    unequal size are handled exactly like the elite branch of "randomWidth"
         *    (addOverhangNodes()/deleteOverhangNodes()), so the host grows or shrinks by the size
         *    difference. The replaced nodes are wired into the existing graph and are therefore
         *    active immediately: they carry NO transplantBlockID/isBlockEntry. They start with
         *    used == false and zeroed traversal counters instead of the donor's values: they lay on the
         *    donor's active path but were never traversed in the host, and an inherited used == true
         *    would hide them from the deletion branch of addDelNodes() forever, so never-traversed
         *    transplant nodes would pile up beyond the junk quota. generationReceived together with
         *    crossoverProtection keeps them alive until the next evaluation sets `used` from the host's
         *    real traversal. WARNING: with crossoverProtection == 0 a directly following
         *    callAddDelNodes() can remove the transplant in the same generation.
         * @param currentGeneration Current generation, stored in Node::generationReceived of received nodes.
         * @param crossoverProtection Generations during which received nodes are protected from further
         *  crossover and from deletion in addDelNodes().
         *  @param traversalNeighbor If true, the crossover is only applied to pairs of individuals that are neighbors in the traversal space 
         *  calculated by traverseCounter.
         *  @param lowerBoundTraversalCounter Lower bound for traverseCounter ratio to consider individuals as neighbors (used if traversalNeighbor is true)
         *  @param upperBoundTraversalCounter Upper bound for traverseCounter ratio to consider individuals as neighbors (used if traversalNeighbor is true)
         * @param lineagePriorCap Upper bound on each side's lineage observation count when the lineage
         *  statistics are blended (see Network::blendLineageWith()).
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

            // "semantic": nodeExchangeRate < 0 couples both rates (see docstring)
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

            // ─── "caseSpecialist*": own donor search and insertion, independent of the pairing below ───
            const bool caseSpecialistAppend  = (type == "caseSpecialist" || type == "caseSpecialistAppend");
            const bool caseSpecialistReplace = (type == "caseSpecialistReplace");
            checkStep(caseSpecialistAppend || caseSpecialistReplace ? Step::CaseSpecialist : Step::Crossover,
                      "crossover(type=\"" + type + "\")");

            if(caseSpecialistAppend || caseSpecialistReplace){
                for(int hostIdx = 0; hostIdx < static_cast<int>(individuals.size()); hostIdx++){
                    auto& host = individuals[hostIdx];
                    if(host.caseScores.empty()) continue; // no per-case data yet this generation

                    // Only Replace skips the elite: it destroys host material, Append is fitness-neutral
                    if(caseSpecialistReplace &&
                       std::find(indicesElite.begin(), indicesElite.end(), hostIdx) != indicesElite.end()){
                        continue;
                    }

                    // Per-host gate, drawn before the expensive donor search
                    if(!distributionBernoulli(*generator)) continue;

                    int bestDonorIdx = -1;
                    float bestScore = 0.0f; // must stay > 0 to be accepted (donor must help on >=1 case)
                    std::vector<int> bestDeficitCases;

                    for(int donorIdx = 0; donorIdx < static_cast<int>(individuals.size()); donorIdx++){
                        if(donorIdx == hostIdx) continue;
                        auto& donor = individuals[donorIdx];
                        if(donor.caseScores.size() != host.caseScores.size()) continue; // case batches must line up
                        if(donor.visitedNodesPerCase.size() != donor.caseScores.size()) continue; // tracking not (yet) populated

                        float score = 0.0f;
                        std::vector<int> deficitCases;
                        for(size_t s = 0; s < host.caseScores.size(); s++){
                            float diff = donor.caseScores[s] - host.caseScores[s];
                            if(diff > 0.0f){
                                score += diff;
                                deficitCases.push_back(static_cast<int>(s));
                            }
                        }

                        if(score > bestScore){
                            bestScore = score;
                            bestDonorIdx = donorIdx;
                            bestDeficitCases = std::move(deficitCases);
                        }
                    }

                    if(bestDonorIdx == -1) continue; // no individual improves on any case -> nothing to transplant

                    auto& donor = individuals[bestDonorIdx];

                    // Union of nodes the donor actually traversed on its deficit cases -- the donor's
                    // "active sub-graph" for exactly the cases where it outperforms the host.
                    std::unordered_set<int> subGraphSet;
                    for(int s : bestDeficitCases){
                        for(int n : donor.visitedNodesPerCase[s]){
                            if(n >= 0 && n < static_cast<int>(donor.innerNodes.size())) subGraphSet.insert(n);
                        }
                    }
                    if(subGraphSet.empty()) continue;

                    std::vector<int> subGraphNodes(subGraphSet.begin(), subGraphSet.end());
                    std::sort(subGraphNodes.begin(), subGraphNodes.end());

                    // ─── "caseSpecialistReplace": overwrite the host's deficit-exclusive sub-graph ───
                    if(caseSpecialistReplace){
                        if(host.visitedNodesPerCase.size() != host.caseScores.size()) continue;

                        std::vector<char> isDeficitCase(host.caseScores.size(), 0);
                        for(int s : bestDeficitCases) isDeficitCase[s] = 1;

                        std::unordered_set<int> hostDeficitNodes, hostOtherNodes;
                        for(size_t s = 0; s < host.visitedNodesPerCase.size(); s++){
                            auto& target = isDeficitCase[s] ? hostDeficitNodes : hostOtherNodes;
                            for(int n : host.visitedNodesPerCase[s]){
                                if(n >= 0 && n < static_cast<int>(host.innerNodes.size())) target.insert(n);
                            }
                        }

                        std::vector<int> hostBlock;
                        for(int n : hostDeficitNodes){
                            if(hostOtherNodes.find(n) == hostOtherNodes.end()) hostBlock.push_back(n);
                        }
                        if(hostBlock.empty()) continue; // no deficit-exclusive structure
                        std::sort(hostBlock.begin(), hostBlock.end()); // required by add/deleteOverhangNodes()

                        // From here like the elite branch of "randomWidth"; the donor only hands out copies
                        const size_t minSubNodes = std::min(subGraphNodes.size(), hostBlock.size());
                        std::unordered_map<int, int> replaceMap =
                            initNodeSwapMap(subGraphNodes, hostBlock, static_cast<int>(host.innerNodes.size()));

                        for(size_t j = 0; j < minSubNodes; j++){
                            auto& node = host.innerNodes[hostBlock[j]];
                            node = donor.innerNodes[subGraphNodes[j]];
                            node.generationReceived = currentGeneration;
                            // Wired in actively: no block tag, so changeEdge() does not restrict access
                            node.transplantBlockID = -1;
                            node.isBlockEntry = false;
                            node.frozen = 0;
                            // Never traversed in the host yet; keeps it deletable (see docstring)
                            node.used = false;
                            // donor's counters would distort the host's statistics
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
                                // As above: addOverhangNodes() copies the donor's used flag, reset it
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

                        host.nCrossovers += 1;
                        host.structureChangedThisGen = true;
                        host.blendLineageWith(donor, lineagePriorCap); // host only, donor unchanged
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
                    // so it is guaranteed to be part of every case's visited set -- this is the single
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

            // Symmetric blend: both read the original values, so call order cannot matter
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
                    // ─── Role-based crossover: homology from node content (see docstring) ───
                    if(!distributionBernoulli(*generator)) continue; // gate per PAIR (= propability)

                    auto meanFitnessOf = [](const Network& net) -> float {
                        if(net.caseScores.empty()) return net.fitness;
                        float sum = 0.0f;
                        for(float v : net.caseScores) sum += v;
                        return sum / static_cast<float>(net.caseScores.size());
                    };

                    // Elite only donates; otherwise the fitter parent donates and stays unchanged
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

                    // Role key as string, usable as map key (groups are small)
                    auto roleKey = [](const Node& node) -> std::string {
                        return node.type + "|" + std::to_string(node.f)
                                         + "|" + std::to_string(node.edges.size());
                    };

                    // Mean deviation of the INNER boundaries (outer ones are the feature range),
                    // normalised by the range; P nodes have none -> 0
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

                    // Candidates per role; matchOnlyUsed drops nodes not entered last evaluation
                    std::unordered_map<std::string, std::vector<int>> donorByRole, recipientByRole;
                    for(size_t k = 0; k < donorNet->innerNodes.size(); k++){
                        if(matchOnlyUsed && !donorNet->innerNodes[k].used) continue;
                        donorByRole[roleKey(donorNet->innerNodes[k])].push_back(static_cast<int>(k));
                    }
                    for(size_t k = 0; k < recipient->innerNodes.size(); k++){
                        if(matchOnlyUsed && !recipient->innerNodes[k].used) continue;
                        recipientByRole[roleKey(recipient->innerNodes[k])].push_back(static_cast<int>(k));
                    }

                    // Donor -> recipient table: greedy by distance within each role (see docstring)
                    std::unordered_map<int, int> donorToRecipient;
                    for(auto const& [role, donorIdxs] : donorByRole){
                        auto it = recipientByRole.find(role);
                        if(it == recipientByRole.end()) continue; // role only in donor -> disjoint
                        const std::vector<int>& recipientIdxs = it->second;

                        struct Candidate { float distance; int donorIdx; int recipientIdx; };
                        std::vector<Candidate> candidates;
                        candidates.reserve(donorIdxs.size() * recipientIdxs.size());
                        for(int d : donorIdxs){
                            for(int r : recipientIdxs){
                                float distance = boundaryDistance(donorNet->innerNodes[d],
                                                                  recipient->innerNodes[r]);
                                // Same role, but cuts too far apart: not the same gene
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

                    if(donorToRecipient.empty()) continue; // no shared role -> pair unchanged
                    crossoverNodesMatched += static_cast<int>(donorToRecipient.size());

                    // All-or-nothing per node: taken only if EVERY edge translates (see docstring)
                    bool changed = false;
                    for(auto const& [donorIdx, recipientIdx] : donorToRecipient){
                        if(!distributionNodeExchange(*generator)) continue;

                        const Node& donorNode = donorNet->innerNodes[donorIdx];

                        // Translate first; the recipient is untouched until the whole gene fits
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
                        targetNode.id = keptPosition;              // position belongs to the recipient
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

                    // An elite only donates; a single unused node is nothing to donate, and the
                    // bidirectional swap below would change the elite
                    if((parent1IsElite && successor1.size() <= 1) || (parent2IsElite && successor2.size() <= 1)) continue;

                    if(parent1IsElite){
                        // parent1 is elite → only parent2 receives genes
                        parent2.nCrossovers += 1;
                        parent2.structureChangedThisGen = true;
                        // Elite stays unchanged; only the recipient blends lineage stats
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

                    } else if(parent2IsElite){
                        // parent2 is elite → only parent1 receives genes
                        parent1.nCrossovers += 1;
                        parent1.structureChangedThisGen = true;
                        // Elite stays unchanged; only the recipient blends lineage stats
                        parent1.blendLineageWith(parent2, lineagePriorCap);

                        int minSubNodes = std::min(successor1.size(), successor2.size());

                        std::unordered_map<int, int> swapMap2 = initNodeSwapMap(successor2, successor1, parent1.innerNodes.size());

                        // Copy nodes from parent2 to parent1
                        for(int j=0; j<minSubNodes; j++){
                            parent1.innerNodes[successor1[j]] = parent2.innerNodes[successor2[j]];
                            parent1.innerNodes[successor1[j]].generationReceived = currentGeneration;
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

                    // Blend lineage along the gene flow and count once per pair, like "randomWidth"
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
         * @details Starting from @p subNodesStart, this method collects successor nodes. By default,
         * nodes entered AFTER the start node (Node::lastVisitStep greater) are successors. Two separate
         * criteria use two separate fields: the ordering along the path comes from the timestamp
         * lastVisitStep, the usage similarity of traversalNeighbor from the frequency traverseCounter
         * (traverseCounter used to carry the timestamp, so the ratio criterion compared timestamps).
         * Nodes of the start node's frozen block (frozen > 0) are always included. If the start node is
         * unused, only the start node (plus its frozen block) is returned. The resulting indices are
         * sorted in ascending order.
         *
         * @param individual The individual (network) whose inner nodes are traversed.
         * @param subNodesStart Index of the starting node in @c individual.innerNodes.
         *                      If -1, a random index is selected uniformly from the available inner nodes.
         * @param nSelectedNodes Maximum number of successor nodes to collect (excluding the start node).
         *                       If -1, a random count between 1 and the number of inner nodes minus one is chosen.
         * @param traversalNeighbor If true, collects nodes whose traverseCounter lies within
         *                          [lowerBound, upperBound] times the start node's, instead of later nodes.
         * @param graphDepth If != -1, collects nodes by breadth-first search along the edges up to this depth instead.
         * @param lowerBoundTraversalCounter Lower ratio bound for traversalNeighbor (default 0.9).
         * @param upperBoundTraversalCounter Upper ratio bound for traversalNeighbor (default 1.1).
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


            // Ordering from lastVisitStep, usage similarity from traverseCounter (see docstring)
            int lastVisitStepStart = individual.innerNodes[subNodesStart].lastVisitStep;
            int traverseCounterStart = individual.innerNodes[subNodesStart].traverseCounter;
            int lastFrozenValue = individual.innerNodes[subNodesStart].frozen;
            for(int i=0; i<individual.innerNodes.size(); i++){
                int lastVisitStepNode = individual.innerNodes[i].lastVisitStep;
                int traverseCounterNode = individual.innerNodes[i].traverseCounter;
                int frozenValueNode = individual.innerNodes[i].frozen;
                // Keeps frozen blocks together; frozen > 0 required, else it is always true
                // (frozen == 0 everywhere) and "randomWidth" swaps an arbitrary index range
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
         * - Removes unused nodes down to the junk quota (see Network::addDelNodes())
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
         *  If you use multiple traversal path per generation, e.g. evaluating multiple cases, elite protection is not garanteed and 
         *  noElite should be set to true. Default is false.
         * @param currentGeneration Current generation (see Network::addDelNodes()).
         * @param crossoverProtection Generations during which nodes received by crossover are not deleted.
         * @param nodeGracePeriod Grace period for recently active nodes (see Network::addDelNodes()); -1 disables it.
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
            checkStep(Step::AddDelNodes, "callAddDelNodes()");

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
            checkStep(Step::AgeUnusedNodes, "callAgeUnusedNodes()");
            for(auto& network : individuals){
                network.ageUnusedNodes();
            }
        }

        /**
         * @brief Calculates Pareto objectives (mean score, minimum score) from caseScores.
         *
         * @details
         * For each individual, extracts two objectives from its case scores:
         *   - objectives[0] = mean score across all cases
         *   - objectives[1] = minimum score across all cases (worst case)
         *
         * The scalar fitness is set to the mean score.
         *
         * @pre The individuals have been evaluated, so caseScores is filled.
         */
        void calculateParetoObjectives(){
            checkStep(Step::Evaluation, "calculateParetoObjectives()");
            for(auto& network : individuals){
                if(network.caseScores.empty()) continue;

                float totalScore = 0.0f;
                float minScore = std::numeric_limits<float>::max();
                for(float value : network.caseScores){
                    totalScore += value;
                    if(value < minScore){
                        minScore = value;
                    }
                }
                float meanScore = totalScore / static_cast<float>(network.caseScores.size());

                network.objectives = {meanScore, minScore};
                network.fitness = meanScore;
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
         * Each tournament winner is drawn at random from the tournament's non-dominated
         * individuals (by Network::objectives). Elitism uses a single elite group:
         * The E best individuals are selected by their Euclidean distance
         * to the Utopia point (the vector of per-objective maxima across
         * the entire population). This preserves individuals that are
         * well-rounded across all objectives.
         *
         * @param N Tournament size
         * @param E Number of elite individuals by Utopia distance
         */
        void paretoTournamentSelection(int N, int E){
            checkStep(Step::Selection, "paretoTournamentSelection()");
            std::vector<Network> selection;
            selection.reserve(individuals.size());
            std::uniform_int_distribution<int> distribution(0, individuals.size()-1);
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
            }

            // Utopia-point elitism
            setEliteUtopia(E, individuals, selection);
            individuals = std::move(selection);
            for (int i = 0; i < static_cast<int>(individuals.size()); i++) {
                bool isElite = std::find(indicesElite.begin(), indicesElite.end(), i) != indicesElite.end();
                individuals[i].frozenExperience = isElite;
            }
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
         * @param individuals Current population (nBest of each chosen elite is incremented)
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
                added++;
            }
        }

        /**
         * @brief Applies gamma mutation to all non-elite JE nodes in the population.
         * @param probability Probability that each node's gamma is resampled.
         * @param justUsedNodes If true, only mutates nodes that were used during traversal.
         */
        void callGammaMutation(float probability, bool justUsedNodes = false) {
            checkStep(Step::Mutation, "callGammaMutation()");
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
            checkStep(Step::Mutation, "callAlphaMutation()");
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
