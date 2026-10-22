#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>
#include "../include/Network.hpp"
#include "../include/Population.hpp"
#include "../include/GymnasiumWrapper.hpp"
#include <pybind11/stl_bind.h>

namespace py = pybind11;

// Opaque so pop.individuals is accessed by reference instead of deep-copying every Network.
PYBIND11_MAKE_OPAQUE(std::vector<Network>)

// Fills a reused buffer from a 2D float32 array to avoid allocation churn across generations.
static void fill_vec2d_from_numpy(
        py::array_t<float, py::array::c_style | py::array::forcecast> &X,
        std::vector<std::vector<float>> &vec2d) {
    py::buffer_info buf = X.request();
    if (buf.ndim != 2)
        throw std::runtime_error("X must be a 2D array");
    size_t nrows = static_cast<size_t>(buf.shape[0]);
    size_t ncols = static_cast<size_t>(buf.shape[1]);
    float* ptr = static_cast<float*>(buf.ptr);
    vec2d.resize(nrows);
    for (size_t i = 0; i < nrows; ++i) {
        vec2d[i].assign(ptr + i * ncols, ptr + (i + 1) * ncols);
    }
}

// Reclaims cyclic garbage left by many env.step()/env.reset() calls.
static void force_gc_collect() {
    py::module_::import("gc").attr("collect")();
}

PYBIND11_MODULE(_core, m) {

    // EdgeExperience
    py::class_<Node::EdgeExperience>(m, "EdgeExperience")
        .def(py::init<>())
        .def_readwrite("meanObs",    &Node::EdgeExperience::meanObs)
        .def_readwrite("m2Obs",      &Node::EdgeExperience::m2Obs)
        .def_readwrite("meanReturn", &Node::EdgeExperience::meanReturn)
        .def_readwrite("m2Return",   &Node::EdgeExperience::m2Return)
        .def_readwrite("n",          &Node::EdgeExperience::n)
        .def(py::pickle(
            [](const Node::EdgeExperience &e) {
                return py::make_tuple(e.meanObs, e.m2Obs, e.meanReturn, e.m2Return, e.n);
            },
            [](py::tuple t) {
                if (t.size() != 5)
                    throw std::runtime_error("Invalid state for EdgeExperience!");
                Node::EdgeExperience e;
                e.meanObs    = t[0].cast<float>();
                e.m2Obs      = t[1].cast<float>();
                e.meanReturn = t[2].cast<float>();
                e.m2Return   = t[3].cast<float>();
                e.n          = t[4].cast<int>();
                return e;
            }
        ));

    // Node
    py::class_<Node>(m, "Node")
    .def(py::init<
            std::shared_ptr<std::mt19937_64>,
            unsigned int,
            std::string,
            unsigned int>(),
         py::arg("generator"), py::arg("id"), py::arg("type"), py::arg("f"))
    .def("rngPointer", &Node::rngPointer)   // diagnostic: shared-RNG identity
    .def_readwrite("id", &Node::id)
    .def_readwrite("type", &Node::type)
    .def_readwrite("f", &Node::f)
    .def_readwrite("edges", &Node::edges)
    .def_readwrite("boundaries", &Node::boundaries)
    .def_readwrite("productionRuleParameter", &Node::productionRuleParameter)
    .def_readwrite("k_d", &Node::k_d)
    .def_readwrite("used", &Node::used)
    .def_readwrite("frozen", &Node::frozen)
    .def_readwrite("traverseCounter", &Node::traverseCounter)
    .def_readwrite("lastVisitStep", &Node::lastVisitStep)
    .def_readwrite("generationReceived", &Node::generationReceived)
    .def_readwrite("everUsed", &Node::everUsed)         // grace period (addDelNodes)
    .def_readwrite("unusedSince", &Node::unusedSince)   // grace period (addDelNodes)
    .def_readwrite("transplantBlockID", &Node::transplantBlockID)   // Population::crossover(type="caseSpecialist")
    .def_readwrite("isBlockEntry", &Node::isBlockEntry)             // Population::crossover(type="caseSpecialist")
    .def_readwrite("edgeExperience", &Node::edgeExperience)
    .def_readwrite("gamma", &Node::gamma)
    .def_readwrite("alpha", &Node::alpha)
    .def("initEdgeExperience",    &Node::initEdgeExperience)
    .def("gammaMutation",         &Node::gammaMutation,  py::arg("probability"))
    .def("alphaMutation",         &Node::alphaMutation,  py::arg("probability"))
    // pickle support
        .def(py::pickle(
        [](const Node &n) { // __getstate__
            return py::make_tuple(
                n.id,
                n.type,
                n.f,
                n.edges,
                n.boundaries,
                n.productionRuleParameter,
                n.k_d,
                n.used,
                n.gamma,
                n.alpha,
                n.edgeExperience,
                n.everUsed,       // grace period (addDelNodes)
                n.unusedSince     // grace period (addDelNodes)
            );
        },
        [](py::tuple t) { // __setstate__
            // 11 = legacy pickles without grace-period fields, kept loadable for archived runs.
            if (t.size() != 11 && t.size() != 13)
                throw std::runtime_error("Invalid state for Node!");

            Node n(
                std::make_shared<std::mt19937_64>(std::random_device{}()),
                t[0].cast<unsigned int>(),
                t[1].cast<std::string>(),
                t[2].cast<unsigned int>()
            );

            n.edges                 = t[3].cast<std::vector<int>>();
            n.boundaries            = t[4].cast<std::vector<double>>();
            n.productionRuleParameter = t[5].cast<std::vector<float>>();
            n.k_d                   = t[6].cast<std::pair<int, int>>();
            n.used                  = t[7].cast<bool>();
            n.gamma                 = t[8].cast<float>();
            n.alpha                 = t[9].cast<float>();
            n.edgeExperience        = t[10].cast<std::vector<Node::EdgeExperience>>();
            if (t.size() == 13) {
                n.everUsed    = t[11].cast<bool>();
                n.unusedSince = t[12].cast<int>();
            }

            return n;
        }
    ));
    // Network
    py::class_<Network>(m, "Network")
    .def(py::init<
            std::shared_ptr<std::mt19937_64>,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            bool>(),
         py::arg("generator"), py::arg("jn"), py::arg("jnf"),
         py::arg("pn"), py::arg("pnf"), py::arg("fractalJudgment"))
    .def_readwrite("jn", &Network::jn)
    .def_readwrite("jnf", &Network::jnf)
    .def_readwrite("pn", &Network::pn)
    .def_readwrite("pnf", &Network::pnf)
    .def_readwrite("fractalJudgment", &Network::fractalJudgment)
    .def("rngPointer", &Network::rngPointer)   // diagnostic: shared-RNG identity
    .def_readwrite("innerNodes", &Network::innerNodes)
    .def_readwrite("startNode", &Network::startNode)
    .def_readwrite("fitness", &Network::fitness)
    .def_readwrite("caseScores", &Network::caseScores)       // score per case of the last evaluation
    .def_readwrite("visitedNodesPerCase", &Network::visitedNodesPerCase) // per-case active sub-graph (caseSpecialist crossover)
    .def_readwrite("objectives", &Network::objectives)       // Pareto objectives
    .def_readwrite("lexicaseObjectives", &Network::lexicaseObjectives)   // lexicase: objectives averaged over the cases
    .def_readwrite("objectivesPerCase", &Network::objectivesPerCase)     // lexicase: objectives per case (type="objectivesPerCase")
    .def_readwrite("decisions", &Network::decisions)
    .def_readwrite("currentNodeID", &Network::currentNodeID)
    .def_readwrite("invalid", &Network::invalid)
    .def_readwrite("nBest", &Network::nBest)
    .def_readwrite("nConsecutiveP", &Network::nConsecutiveP)
    .def_readwrite("nCrossovers", &Network::nCrossovers)
    .def_readwrite("frozenExperience", &Network::frozenExperience)
    .def_readwrite("lineageMean", &Network::lineageMean)
    .def_readwrite("lineageN", &Network::lineageN)
    .def_readwrite("lineageM2", &Network::lineageM2)
    .def_readwrite("lineageSuccessRate", &Network::lineageSuccessRate)
    .def_readwrite("lineageSuccessN", &Network::lineageSuccessN)
    .def_readwrite("structureChangedThisGen", &Network::structureChangedThisGen)
    .def("lineageVariance", &Network::lineageVariance)
    .def("lineageLCB", &Network::lineageLCB, py::arg("z"))
    .def("updateLineageStats", &Network::updateLineageStats, py::arg("rawFitness"), py::arg("success"))
    .def("aggregateCaseResults", &Network::aggregateCaseResults)
    .def("blendLineageWith", &Network::blendLineageWith, py::arg("other"), py::arg("priorCap"))
    .def("initPathTraversal", &Network::initPathTraversal, py::arg("startingFitness")=0.0f, py::arg("newRun")=true)
    .def("decisionAndNextNode",
        [](Network &self, std::vector<double> obs, int dMax) -> int {
            return self.decisionAndNextNode(obs, dMax);
        },
    py::arg("obs"), py::arg("dMax"))
    .def("act",
        [](Network &self, std::vector<double> obs, int dMax, int maxConsecutiveP, int fallbackAction, int holdAction) -> int {
            return self.act(obs, dMax, maxConsecutiveP, fallbackAction, holdAction);
        },
    py::arg("obs"), py::arg("dMax"), py::arg("maxConsecutiveP"), py::arg("fallbackAction")=0, py::arg("holdAction")=-1)
    .def("recordStep", &Network::recordStep, py::arg("reward"))
    .def("traversePath",
        [](Network &self, py::array_t<float, py::array::c_style | py::array::forcecast> X, int dMax) {
            thread_local std::vector<std::vector<float>> vec2d;
            fill_vec2d_from_numpy(X, vec2d);
            {
                py::gil_scoped_release release;
                self.traversePath(vec2d, dMax);
            }
        },
        py::arg("X"), py::arg("dMax"))
    .def("clearUsedNodes", &Network::clearUsedNodes)
    .def("updateExperienceFromEpisode", &Network::updateExperienceFromEpisode)
    // pickle support (11-element tuple)
    .def(py::pickle(
        [](const Network &n) { // __getstate__
            return py::make_tuple(
                n.jn, n.jnf, n.pn, n.pnf, n.fractalJudgment,
                n.innerNodes, n.startNode, n.fitness, n.decisions,
                n.caseScores, n.objectives
            );
        },
        [](py::tuple t) { // __setstate__
            // 12 = pickles before 3.0.0 (trailing lastStepRewards, ignored), kept loadable for archived runs
            if (t.size() != 11 && t.size() != 12)
                throw std::runtime_error("Invalid state for Network!");

            Network net(
                std::make_shared<std::mt19937_64>(std::random_device{}()),
                t[0].cast<unsigned int>(),
                t[1].cast<unsigned int>(),
                t[2].cast<unsigned int>(),
                t[3].cast<unsigned int>(),
                t[4].cast<bool>()
            );

            net.innerNodes = t[5].cast<std::vector<Node>>();
            net.startNode = t[6].cast<Node>();
            net.fitness = t[7].cast<float>();
            net.decisions = t[8].cast<std::vector<int>>();
            net.caseScores = t[9].cast<std::vector<float>>();
            net.objectives = t[10].cast<std::vector<float>>();

            return net;
        }
    ));

    // List-like access by reference; pickles as a plain list of Networks.
    py::bind_vector<std::vector<Network>>(m, "NetworkVector")
        .def(py::pickle(
            [](const std::vector<Network> &v) { // __getstate__
                py::list l;
                for (const auto& n : v)
                    l.append(py::cast(n));
                return l;
            },
            [](py::list l) { // __setstate__
                std::vector<Network> v;
                v.reserve(l.size());
                for (auto item : l)
                    v.push_back(item.cast<Network>());
                return v;
            }
        ));

    // Handed to the episodeObjectives hook of Population.gymnasium()
    py::class_<EpisodeResult>(m, "EpisodeResult")
        .def_readonly("seed", &EpisodeResult::seed)
        .def_readonly("initialObservation", &EpisodeResult::initialObservation)
        .def_readonly("observation", &EpisodeResult::observation)
        .def_readonly("info", &EpisodeResult::info)
        .def_readonly("rewards", &EpisodeResult::rewards)
        .def_readonly("episodeReturn", &EpisodeResult::episodeReturn)
        .def_readonly("steps", &EpisodeResult::steps)
        .def_readonly("terminated", &EpisodeResult::terminated)
        .def_readonly("truncated", &EpisodeResult::truncated)
        .def_readonly("invalid", &EpisodeResult::invalid)
        .def("__repr__", [](const EpisodeResult& e) {
            return "<EpisodeResult seed=" + std::to_string(e.seed) + " steps=" + std::to_string(e.steps) +
                   " return=" + std::to_string(e.episodeReturn) + ">";
        });

    // Population
    py::class_<Population>(m, "Population")
        // Member
        .def(py::init<
                int,
                const unsigned int,
                unsigned int,
                unsigned int,
                unsigned int,
                unsigned int,
                bool,
                bool,
                std::vector<int>
                >(),
             py::arg("seed"), py::arg("ni"), py::arg("jn"), py::arg("jnf"),
             py::arg("pn"), py::arg("pnf"), py::arg("fractalJudgment"), py::arg("useExperience") = false, py::arg("nFeatureValues"))
        .def("rngState", &Population::rngState)     // diagnostic: full RNG state
        .def("rngPointer", &Population::rngPointer) // diagnostic: shared-RNG identity
        .def_readonly("ni", &Population::ni)
        .def_readwrite("jn", &Population::jn)
        .def_readwrite("jnf", &Population::jnf)
        .def_readwrite("pn", &Population::pn)
        .def_readwrite("pnf", &Population::pnf)
        .def_readwrite("fractalJudgment", &Population::fractalJudgment)
        .def_readwrite("useExperience", &Population::useExperience)
        // Fitness statistics are computed from the current individuals on every read
        .def_property_readonly("bestFit", &Population::bestFit)
        .def_property_readonly("meanFitness", &Population::meanFitness)
        .def_property_readonly("minFitness", &Population::minFitness)
        .def_readwrite("indicesElite", &Population::indicesElite)
        .def_readwrite("maxNetworkSize", &Population::maxNetworkSize)
        .def_readwrite("nextTransplantBlockID", &Population::nextTransplantBlockID) // caseSpecialist crossover block-id counter
        // Crossover diagnostics of the last crossover() call (filled by type="semantic" only)
        .def_readonly("crossoverPairsApplied", &Population::crossoverPairsApplied)
        .def_readonly("crossoverNodesMatched", &Population::crossoverNodesMatched)
        .def_readonly("crossoverNodesExchanged", &Population::crossoverNodesExchanged)
        .def_readonly("crossoverNodesSkipped", &Population::crossoverNodesSkipped)
        // Counted in C++ because reading innerNodes from Python copies the whole node vector.
        .def("networkSizes",
            [](Population &self){
                std::vector<int> sizes;
                sizes.reserve(self.individuals.size());
                for(auto& net : self.individuals) sizes.push_back(static_cast<int>(net.innerNodes.size()));
                return sizes;
            },
            "Number of inner nodes per individual (index-aligned with individuals).")
        .def("unusedNodeCounts",
            [](Population &self){
                std::vector<int> counts;
                counts.reserve(self.individuals.size());
                for(auto& net : self.individuals){
                    int unused = 0;
                    for(auto& node : net.innerNodes) if(!node.used) unused += 1;
                    counts.push_back(unused);
                }
                return counts;
            },
            "Number of nodes never entered during the last evaluation, per individual "
            "(dormant transplant blocks and junk DNA).")
        // Plain reference avoids the keep_alive entry reference_internal adds on every access.
        .def_property("individuals",
            [](Population &self) -> std::vector<Network>& {
                return self.individuals;
            },
            [](Population &self, py::object v) {
                if (py::isinstance<std::vector<Network>>(v)) {
                    // Direct assignment from NetworkVector
                    self.individuals = v.cast<std::vector<Network>&>();
                } else {
                    // Accept any Python sequence of Network objects (e.g. list)
                    py::sequence seq = v.cast<py::sequence>();
                    std::vector<Network> tmp;
                    tmp.reserve(py::len(seq));
                    for (auto item : seq)
                        tmp.push_back(item.cast<Network>());
                    self.individuals = std::move(tmp);
                }
            },

            py::return_value_policy::reference)
        .def_readwrite("nFeatureValues", &Population::nFeatureValues)

        // Functions
        .def(
            "setAllNodeBoundaries",
            [](Population &p, py::list minF_py, py::list maxF_py)
            {
                std::vector<float> minF;
                std::vector<float> maxF;

                for (auto item : minF_py)
                    minF.push_back(item.cast<float>());

                for (auto item : maxF_py)
                    maxF.push_back(item.cast<float>());

                {
                    py::gil_scoped_release release;
                    p.setAllNodeBoundaries(minF, maxF);
                }
            },
            py::arg("minF"),
            py::arg("maxF")
        )

        .def("callTraversePath",
            [](Population &self, py::array_t<float, py::array::c_style | py::array::forcecast> X, int dMax) {
                thread_local std::vector<std::vector<float>> vec2d;
                fill_vec2d_from_numpy(X, vec2d);
                {
                    py::gil_scoped_release release;
                    self.callTraversePath(vec2d, dMax);
                }
            },
            py::arg("X"), py::arg("dMax"))

        .def("accuracy",
            [](Population &self,
               py::array_t<float, py::array::c_style | py::array::forcecast> X,
               py::array_t<int, py::array::c_style | py::array::forcecast> y,
               int dMax, int penalty) {
                thread_local std::vector<std::vector<float>> vec2d;
                fill_vec2d_from_numpy(X, vec2d);

                py::buffer_info ybuf = y.request();
                if (ybuf.ndim != 1)
                    throw std::runtime_error("y must be a 1D array");
                int* yptr = static_cast<int*>(ybuf.ptr);
                std::vector<int> y_vec(yptr, yptr + ybuf.shape[0]);

                {
                    py::gil_scoped_release release;
                    self.accuracy(vec2d, y_vec, dMax, penalty);
                }
            },
            py::arg("X"), py::arg("y"), py::arg("dMax"), py::arg("penalty"))

        .def("gymnasium",
                [](Population &self,
                    py::object env,
                    std::vector<int> seeds,
                    int dMax,
                    int maxSteps,
                    int maxConsecutiveP,
                    int fallbackAction,
                    int holdAction,
                    py::object resetOptions,
                    py::object episodeObjectives,
                    bool trackLineage,
                    float successThreshold
                    ) {
                        EpisodeSettings settings{dMax, maxSteps, maxConsecutiveP, fallbackAction, holdAction};
                        evaluateEpisodes(self.individuals, env, seeds, resetOptions, settings,
                                         episodeObjectives, trackLineage, successThreshold);
                        self.markEvaluated();
                        force_gc_collect();
                    },
                py::arg("env"), py::arg("seeds"), py::arg("dMax"), py::arg("maxSteps"), py::arg("maxConsecutiveP"),
                py::arg("fallbackAction")=0, py::arg("holdAction")=-1,
                py::arg("resetOptions")=py::none(), py::arg("episodeObjectives")=py::none(),
                py::arg("trackLineage")=false,
                py::arg("successThreshold")=std::numeric_limits<float>::infinity(),
                R"doc(Evaluate every individual on one episode per seed (one case per seed).

Args:
    env: Gymnasium environment (reset(seed, options) / step(action) API). Problem-specific
        reward shaping or start-state changes belong into a gymnasium.Wrapper around it.
    seeds: Seeds for env.reset(); negative = unseeded.
    dMax: Maximum consecutive judgment nodes per decision.
    maxSteps: Episode length limit.
    maxConsecutiveP: Maximum consecutive processing nodes before fallbackAction is used.
    fallbackAction: Action executed for an invalid network (default 0).
    holdAction: Processing function that repeats the previous decision; -1 = off.
    resetOptions: None or one env.reset(options=...) entry per seed.
    episodeObjectives: None or callable(EpisodeResult) -> list of floats (larger = better),
        stored per case in objectivesPerCase and averaged into lexicaseObjectives.
    trackLineage: Update the lineage statistics with every case.
    successThreshold: A case above this score counts as a success for lineageSuccessRate.

Writes fitness (mean return), caseScores, visitedNodesPerCase, objectivesPerCase and
lexicaseObjectives of every individual.)doc"
            )

        .def_readwrite("checkOrder", &Population::checkOrder,
             "If True (default), a step out of the order of a generation raises RuntimeError: "
             "evaluation, callAgeUnusedNodes, selection, updateAdaptiveK, crossover caseSpecialist*, "
             "callAddDelNodes, crossover, mutations, capLineageAfterMutation. Set to False to call "
             "single operators on purpose.")
        .def("markEvaluated", &Population::markEvaluated,
             "Mark the population as freshly evaluated, so the order check starts a new generation. "
             "gymnasium(), callTraversePath() and accuracy() do this themselves; call it after an "
             "evaluation outside of them (ParallelEvaluator calls it for you).")

        .def("calculateParetoObjectives", &Population::calculateParetoObjectives,
             py::call_guard<py::gil_scoped_release>())

        .def("paretoTournamentSelection", &Population::paretoTournamentSelection,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("N"), py::arg("E"))

        .def("tournamentSelection", &Population::tournamentSelection,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("N"), py::arg("E"), py::arg("useLineageFitness")=false, py::arg("minLineageN")=10, py::arg("lineageZ")=0.0f)
        .def("lexicaseSelection", &Population::lexicaseSelection,
             py::call_guard<py::gil_scoped_release>(),
             // epsilon: score margin (<0 = MAD); standard deviations for "casesStandardized"
             // (<0 = 0.5); rank positions for "casesRank" (<0 = 3).
             py::arg("E"), py::arg("epsilon")=-1.0f, py::arg("type")=std::string("objectives"))
        .def("capLineageAfterMutation", &Population::capLineageAfterMutation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("priorCap")=5)
        .def_static("lineageRobustSuccess", &Population::lineageRobustSuccess,
             py::arg("ind"), py::arg("z")=1.0f)
        .def_static("caseConsistentSuccess", &Population::caseConsistentSuccess,
             py::arg("ind"), py::arg("requiredFraction")=1.0f, py::arg("z")=1.0f)
        .def_static("caseThresholdSuccess", &Population::caseThresholdSuccess,
             py::arg("ind"), py::arg("threshold"), py::arg("requiredFraction")=1.0f)
        .def_static("caseConsistencyRatio", &Population::caseConsistencyRatio,
             py::arg("ind"))
        .def("updateAdaptiveK", &Population::updateAdaptiveK,
             py::arg("successCriterion") = Population::SuccessCriterion([](const Network& ind){ return Population::caseConsistencyRatio(ind); }),
             py::arg("useRechenberg")=true,
             py::arg("invert")=false,
             py::arg("windowSize")=10,
             py::arg("targetSuccessRate")=0.2f,
             py::arg("incFactor")=1.22f,
             py::arg("decFactor")=0.82f,
             py::arg("kMin")=0.1f,
             py::arg("kMax")=5.0f)
        .def_readwrite("adaptiveK", &Population::adaptiveK_)
        .def_property_readonly("successRateHistory", [](const Population& p){
             return std::vector<float>(p.successRateHistory_.begin(), p.successRateHistory_.end());
         })
        .def("callEdgeMutation", &Population::callEdgeMutation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probInnerNodes"), py::arg("probStartNode"), py::arg("justUsedNodes")=false, py::arg("k")=0.0f)
        .def("callBoundaryMutationNormal", &Population::callBoundaryMutationNormal,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false, py::arg("k")=0.0f)
        .def("callBoundaryMutationUniform", &Population::callBoundaryMutationUniform,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("justUsedNodes")=false, py::arg("k")=0.0f)
        .def("callBoundaryMutationNetworkSizeDependingSigma", &Population::callBoundaryMutationNetworkSizeDependingSigma,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false, py::arg("k")=0.0f)
        .def("callBoundaryMutationEdgeSizeDependingSigma", &Population::callBoundaryMutationEdgeSizeDependingSigma,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false, py::arg("k")=0.0f)
        .def(
            "callBoundaryMutationFractal",
            [](Population &p, float probability, py::list minF_py, py::list maxF_py, bool justUsedNodes, float k)
            {
                std::vector<float> minF;
                std::vector<float> maxF;

                for (auto item : minF_py)
                    minF.push_back(item.cast<float>());

                for (auto item : maxF_py)
                    maxF.push_back(item.cast<float>());

                {
                    py::gil_scoped_release release;
                    p.callBoundaryMutationFractal(probability, minF, maxF, justUsedNodes, k);
                }
            },
            py::arg("probability"),
            py::arg("minF"),
            py::arg("maxF"),
            py::arg("justUsedNodes")=false,
            py::arg("k")=1.0f
        )

        .def("callGammaMutation", &Population::callGammaMutation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("justUsedNodes") = false)
        .def("callAlphaMutation", &Population::callAlphaMutation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("justUsedNodes") = false)

        .def("crossover", &Population::crossover,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), 
             py::arg("type"), 
             py::arg("currentGeneration")=0,
             py::arg("crossoverProtection") = 0,
             py::arg("traversalNeighbor")=false, 
             py::arg("lowerBoundTraversalCounter")=0.9,
             py::arg("upperBoundTraversalCounter")=1.1,
             py::arg("lineagePriorCap")=5,
             py::arg("boundaryTolerance")=1.0f,   // type="semantic" only
             py::arg("matchOnlyUsed")=false,      // type="semantic" only
             py::arg("nodeExchangeRate")=-1.0f)   // type="semantic" only; <0 = use probability
        .def(
            "callAddDelNodes",
            [](Population &p, py::list minF_py, py::list maxF_py, float junk, bool noElite, int currentGeneration, int crossoverProtection, int nodeGracePeriod)
            {
                std::vector<float> minF;
                std::vector<float> maxF;

                for (auto item : minF_py)
                    minF.push_back(item.cast<float>());

                for (auto item : maxF_py)
                    maxF.push_back(item.cast<float>());

                {
                    py::gil_scoped_release release;
                    p.callAddDelNodes(minF, maxF, junk, noElite, currentGeneration, crossoverProtection, nodeGracePeriod);
                }
            },
            py::arg("minF"),
            py::arg("maxF"),
            py::arg("junk")=0,
            py::arg("noElite")=false,
            py::arg("currentGeneration") = 0,
            py::arg("crossoverProtection") = 3,
            py::arg("nodeGracePeriod") = -1   // -1 disables the grace period
        )
        .def("callAgeUnusedNodes", &Population::callAgeUnusedNodes,
            py::call_guard<py::gil_scoped_release>())

        // pickle support; individuals are stored as a plain Python list
        .def(py::pickle(
        [](const Population &p) { // __getstate__
            py::list ind_list;
            for (const auto& net : p.individuals)
                ind_list.append(py::cast(net));
            return py::make_tuple(
                p.ni, p.jn, p.jnf, p.pn, p.pnf, p.fractalJudgment,
                p.indicesElite, ind_list
            );
        },
        [](py::tuple t) { // __setstate__
            // Older pickles have 11 fields: they also stored bestFit, meanFitness and minFitness
            const bool legacy = t.size() == 11;
            if (t.size() != 8 && !legacy)
                throw std::runtime_error("Invalid state for Population!");

            Population p(
                0, // placeholder for seed (not a member)
                t[0].cast<unsigned int>(),
                t[1].cast<unsigned int>(),
                t[2].cast<unsigned int>(),
                t[3].cast<unsigned int>(),
                t[4].cast<unsigned int>(),
                t[5].cast<bool>()
            );

            p.indicesElite = t[legacy ? 7 : 6].cast<std::vector<int>>();

            p.individuals.clear();
            py::list ind_list = t[legacy ? 10 : 7].cast<py::list>();
            for (auto item : ind_list)
                p.individuals.push_back(item.cast<Network>());

            return p;
        }
    ))


        ;
}
