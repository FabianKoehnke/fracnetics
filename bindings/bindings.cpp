// ---------------------------------------------------------------------------
// Memory-efficient pybind11 bindings for fracnetics
//
// Key design decisions (following PyTorch-style best practices):
//
// 1. PYBIND11_MAKE_OPAQUE(std::vector<Network>)
//    Prevents pybind11 from deep-copying the entire individuals vector on
//    every Python access.  py::bind_vector provides a thin reference wrapper.
//
// 2. def_property (copy semantics) for all complex-type members
//    def_readwrite uses reference_internal which adds keep_alive entries
//    in pybind11's internal patients map on EVERY read.  Since the returned
//    Python objects are copies anyway (pybind11 converts std::vector<T> →
//    Python list via stl.h), the keep_alive is pure overhead.  def_property
//    with explicit copy getter/setter avoids this entirely.
//
// 3. Buffer-protocol numpy access for observation / feature data
//    fill_vec2d_from_numpy() copies data directly from the numpy buffer
//    into a pre-allocated thread_local vec2d, avoiding per-call heap churn.
//    fitGymnasium uses py::array_t with buffer protocol to extract obs.
//
// 4. Periodic gc.collect() inside gymnasium loops
//    gymnasium/gymnasiumMultiSeed inline the population loop so that
//    Python's cyclic GC can be invoked every N individuals.  This keeps
//    peak RSS bounded by collecting cyclic garbage from env.step()/reset()
//    before it accumulates across the entire population.
//
// 5. py::gil_scoped_release on all pure-C++ methods
//    Allows other Python threads to progress and lets the OS reclaim
//    pages while heavy computation runs.
// ---------------------------------------------------------------------------

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

// --- Opaque vector declaration -------------------------------------------
// Must appear before PYBIND11_MODULE so pybind11 treats std::vector<Network>
// as opaque (reference semantics) rather than converting to a Python list
// (deep copy).
PYBIND11_MAKE_OPAQUE(std::vector<Network>)

// --- Helper: numpy → vec2d -----------------------------------------------
// Fills a pre-allocated 2-D float buffer from a contiguous numpy array.
// Uses thread_local storage so the buffer survives across calls without
// being freed and re-allocated each generation.
static void fill_vec2d_from_numpy(
        py::array_t<float, py::array::c_style | py::array::forcecast> &X,
        std::vector<std::vector<float>> &vec2d) {
    py::buffer_info buf = X.request();
    if (buf.ndim != 2)
        throw std::runtime_error("X must be a 2D array");
    size_t nrows = static_cast<size_t>(buf.shape[0]);
    size_t ncols = static_cast<size_t>(buf.shape[1]);
    const float* ptr = static_cast<const float*>(buf.ptr);
    vec2d.resize(nrows);
    for (size_t i = 0; i < nrows; ++i) {
        vec2d[i].assign(ptr + i * ncols, ptr + (i + 1) * ncols);
    }
}

// --- Helper: gc.collect() ------------------------------------------------
static void force_gc_collect() {
    py::module_::import("gc").attr("collect")();
}

// =========================================================================

PYBIND11_MODULE(_core, m) {

    // =====================================================================
    // Node
    // =====================================================================
    py::class_<Node>(m, "Node")
    .def(py::init<
            std::shared_ptr<std::mt19937_64>,
            unsigned int,
            std::string,
            unsigned int>(),
         py::arg("generator"), py::arg("id"), py::arg("type"), py::arg("f"))
    // --- Primitive types: def_readwrite is fine (no keep_alive overhead) --
    .def_readwrite("id", &Node::id)
    .def_readwrite("type", &Node::type)
    .def_readwrite("f", &Node::f)
    .def_readwrite("used", &Node::used)
    .def_readwrite("traverseCounter", &Node::traverseCounter)
    // --- Complex types: def_property with explicit copy -------------------
    // Avoids reference_internal / keep_alive / patients-map accumulation.
    .def_property("edges",
        [](const Node &n) { return n.edges; },
        [](Node &n, const std::vector<int> &v) { n.edges = v; })
    .def_property("boundaries",
        [](const Node &n) { return n.boundaries; },
        [](Node &n, const std::vector<double> &v) { n.boundaries = v; })
    .def_property("productionRuleParameter",
        [](const Node &n) { return n.productionRuleParameter; },
        [](Node &n, const std::vector<float> &v) { n.productionRuleParameter = v; })
    .def_property("k_d",
        [](const Node &n) { return n.k_d; },
        [](Node &n, const std::pair<int,int> &v) { n.k_d = v; })
    // --- Pickle ---
    .def(py::pickle(
        [](const Node &n) {
            return py::make_tuple(
                n.id, n.type, n.f,
                n.edges, n.boundaries, n.productionRuleParameter,
                n.k_d, n.used
            );
        },
        [](py::tuple t) {
            if (t.size() != 8)
                throw std::runtime_error("Invalid state for Node!");
            Node n(
                std::make_shared<std::mt19937_64>(std::random_device{}()),
                t[0].cast<unsigned int>(),
                t[1].cast<std::string>(),
                t[2].cast<unsigned int>()
            );
            n.edges = t[3].cast<std::vector<int>>();
            n.boundaries = t[4].cast<std::vector<double>>();
            n.productionRuleParameter = t[5].cast<std::vector<float>>();
            n.k_d = t[6].cast<std::pair<int, int>>();
            n.used = t[7].cast<bool>();
            return n;
        }
    ));

    // =====================================================================
    // Network
    // =====================================================================
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
    // --- Primitive types (no overhead) ------------------------------------
    .def_readwrite("jn", &Network::jn)
    .def_readwrite("jnf", &Network::jnf)
    .def_readwrite("pn", &Network::pn)
    .def_readwrite("pnf", &Network::pnf)
    .def_readwrite("fractalJudgment", &Network::fractalJudgment)
    .def_readwrite("fitness", &Network::fitness)
    .def_readwrite("currentNodeID", &Network::currentNodeID)
    .def_readwrite("invalid", &Network::invalid)
    .def_readwrite("nBest", &Network::nBest)
    .def_readwrite("nConsecutiveP", &Network::nConsecutiveP)
    .def_readwrite("nCrossovers", &Network::nCrossovers)
    // --- Complex types: def_property with copy ----------------------------
    .def_property("innerNodes",
        [](const Network &n) { return n.innerNodes; },
        [](Network &n, const std::vector<Node> &v) { n.innerNodes = v; })
    .def_property("startNode",
        [](const Network &n) { return n.startNode; },
        [](Network &n, const Node &v) { n.startNode = v; })
    .def_property("fitnessValues",
        [](const Network &n) { return n.fitnessValues; },
        [](Network &n, const std::vector<float> &v) { n.fitnessValues = v; })
    .def_property("objectives",
        [](const Network &n) { return n.objectives; },
        [](Network &n, const std::vector<float> &v) { n.objectives = v; })
    .def_property("lastStepRewards",
        [](const Network &n) { return n.lastStepRewards; },
        [](Network &n, const std::vector<float> &v) { n.lastStepRewards = v; })
    .def_property("decisions",
        [](const Network &n) { return n.decisions; },
        [](Network &n, const std::vector<int> &v) { n.decisions = v; })
    // --- Methods ----------------------------------------------------------
    .def("initPathTraversal", &Network::initPathTraversal, py::arg("startingFitness")=0.0f)
    .def("decisionAndNextNode",
        [](Network &self, std::vector<double> obs, int dMax) -> int {
            return self.decisionAndNextNode(obs, dMax);
        },
        py::arg("obs"), py::arg("dMax"))
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
    // --- Pickle ---
    .def(py::pickle(
        [](const Network &n) {
            return py::make_tuple(
                n.jn, n.jnf, n.pn, n.pnf, n.fractalJudgment,
                n.innerNodes, n.startNode, n.fitness, n.decisions,
                n.fitnessValues, n.objectives, n.lastStepRewards
            );
        },
        [](py::tuple t) {
            if (t.size() != 12)
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
            net.fitnessValues = t[9].cast<std::vector<float>>();
            net.objectives = t[10].cast<std::vector<float>>();
            net.lastStepRewards = t[11].cast<std::vector<float>>();
            return net;
        }
    ));

    // =====================================================================
    // NetworkVector (opaque std::vector<Network>)
    // =====================================================================
    py::bind_vector<std::vector<Network>>(m, "NetworkVector")
        .def(py::pickle(
            [](const std::vector<Network> &v) {
                py::list l;
                for (const auto& n : v)
                    l.append(py::cast(n));
                return l;
            },
            [](py::list l) {
                std::vector<Network> v;
                v.reserve(l.size());
                for (auto item : l)
                    v.push_back(item.cast<Network>());
                return v;
            }
        ));

    // =====================================================================
    // Population
    // =====================================================================
    py::class_<Population>(m, "Population")
        .def(py::init<
                int,
                const unsigned int,
                unsigned int,
                unsigned int,
                unsigned int,
                unsigned int,
                bool,
                std::vector<int>
                >(),
             py::arg("seed"), py::arg("ni"), py::arg("jn"), py::arg("jnf"),
             py::arg("pn"), py::arg("pnf"), py::arg("fractalJudgment"), py::arg("nFeatureValues"))
        // --- Primitive / small members ------------------------------------
        .def_readonly("ni", &Population::ni)
        .def_readwrite("jn", &Population::jn)
        .def_readwrite("jnf", &Population::jnf)
        .def_readwrite("pn", &Population::pn)
        .def_readwrite("pnf", &Population::pnf)
        .def_readwrite("fractalJudgment", &Population::fractalJudgment)
        .def_readwrite("bestFit", &Population::bestFit)
        .def_readwrite("meanFitness", &Population::meanFitness)
        .def_readwrite("minFitness", &Population::minFitness)
        // --- indicesElite: small int vector, use def_property to be safe --
        .def_property("indicesElite",
            [](const Population &p) { return p.indicesElite; },
            [](Population &p, const std::vector<int> &v) { p.indicesElite = v; })
        // --- individuals: reference to opaque vector (no deep copy) -------
        .def_property("individuals",
            [](Population &self) -> std::vector<Network>& {
                return self.individuals;
            },
            [](Population &self, const std::vector<Network> &v) {
                self.individuals = v;
            },
            py::return_value_policy::reference)
        .def_readwrite("nFeatureValues", &Population::nFeatureValues)

        // =================================================================
        // Functions
        // =================================================================
        .def(
            "setAllNodeBoundaries",
            [](Population &p, py::list minF_py, py::list maxF_py)
            {
                std::vector<float> minF;
                std::vector<float> maxF;
                minF.reserve(minF_py.size());
                maxF.reserve(maxF_py.size());
                for (auto item : minF_py) minF.push_back(item.cast<float>());
                for (auto item : maxF_py) maxF.push_back(item.cast<float>());
                {
                    py::gil_scoped_release release;
                    p.setAllNodeBoundaries(minF, maxF);
                }
            },
            py::arg("minF"), py::arg("maxF")
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

        // -----------------------------------------------------------------
        // gymnasium – inlined loop with periodic gc.collect()
        // -----------------------------------------------------------------
        .def("gymnasium",
                [](Population &self,
                    py::object env,
                    int dMax,
                    int maxSteps,
                    int maxConsecutiveP,
                    int worstFitness,
                    int seed
                    ) {
                        GymEnvWrapper wrapper(env);
                        auto gc_collect = py::module_::import("gc").attr("collect");
                        size_t n = self.individuals.size();
                        for (size_t i = 0; i < n; ++i) {
                            self.individuals[i].fitGymnasium(
                                wrapper, dMax, maxSteps, maxConsecutiveP, worstFitness, seed);
                            // Collect cyclic garbage every 20 individuals to
                            // prevent RSS from growing across the population.
                            if ((i + 1) % 20 == 0) gc_collect();
                        }
                        gc_collect();
                    },
                py::arg("env"), py::arg("dMax"), py::arg("maxSteps"),
                py::arg("maxConsecutiveP"), py::arg("worstFitness"), py::arg("seed")
            )

        // -----------------------------------------------------------------
        // gymnasiumMultiSeed – inlined loop with periodic gc.collect()
        // -----------------------------------------------------------------
        .def("gymnasiumMultiSeed",
                [](Population &self,
                    py::object env,
                    int dMax,
                    int maxSteps,
                    int maxConsecutiveP,
                    int worstFitness,
                    std::vector<int> seeds
                    ) {
                        GymEnvWrapper wrapper(env);
                        auto gc_collect = py::module_::import("gc").attr("collect");
                        size_t n = self.individuals.size();
                        for (size_t i = 0; i < n; ++i) {
                            auto& network = self.individuals[i];
                            network.fitnessValues.clear();
                            network.lastStepRewards.clear();
                            float totalReward = 0.0f;
                            for (int s : seeds) {
                                network.fitGymnasium(
                                    wrapper, dMax, maxSteps, maxConsecutiveP, worstFitness, s);
                                network.fitnessValues.push_back(network.fitness);
                                network.lastStepRewards.push_back(network.lastFitness);
                                totalReward += network.fitness;
                            }
                            network.fitness = totalReward / static_cast<float>(seeds.size());
                            // Collect cyclic garbage every 20 individuals
                            if ((i + 1) % 20 == 0) gc_collect();
                        }
                        gc_collect();
                    },
                py::arg("env"), py::arg("dMax"), py::arg("maxSteps"),
                py::arg("maxConsecutiveP"), py::arg("worstFitness"), py::arg("seeds")
            )

        .def("calculateParetoObjectives", &Population::calculateParetoObjectives,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("landingThreshold")=100.0f)

        .def("paretoTournamentSelection", &Population::paretoTournamentSelection,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("N"), py::arg("E_reward"), py::arg("E_landing"))

        .def("tournamentSelection", &Population::tournamentSelection,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("N"), py::arg("E"))

        .def("callEdgeMutation", &Population::callEdgeMutation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probInnerNodes"), py::arg("probStartNode"),
             py::arg("justUsedNodes")=false, py::arg("adaptToEdgeSize")=false)

        .def("callBoundaryMutationNormal", &Population::callBoundaryMutationNormal,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false)
        .def("callBoundaryMutationUniform", &Population::callBoundaryMutationUniform,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("justUsedNodes")=false)
        .def("callBoundaryMutationNetworkSizeDependingSigma", &Population::callBoundaryMutationNetworkSizeDependingSigma,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false)
        .def("callBoundaryMutationEdgeSizeDependingSigma", &Population::callBoundaryMutationEdgeSizeDependingSigma,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("sigma"), py::arg("justUsedNodes")=false)

        .def(
            "callBoundaryMutationFractal",
            [](Population &p, float probability, py::list minF_py, py::list maxF_py, bool justUsedNodes)
            {
                std::vector<float> minF;
                std::vector<float> maxF;
                minF.reserve(minF_py.size());
                maxF.reserve(maxF_py.size());
                for (auto item : minF_py) minF.push_back(item.cast<float>());
                for (auto item : maxF_py) maxF.push_back(item.cast<float>());
                {
                    py::gil_scoped_release release;
                    p.callBoundaryMutationFractal(probability, minF, maxF, justUsedNodes);
                }
            },
            py::arg("probability"), py::arg("minF"), py::arg("maxF"),
            py::arg("justUsedNodes")=false
        )

        .def("crossover", &Population::crossover,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("probability"), py::arg("type"))

        .def(
            "callAddDelNodes",
            [](Population &p, py::list minF_py, py::list maxF_py, float junk)
            {
                std::vector<float> minF;
                std::vector<float> maxF;
                minF.reserve(minF_py.size());
                maxF.reserve(maxF_py.size());
                for (auto item : minF_py) minF.push_back(item.cast<float>());
                for (auto item : maxF_py) maxF.push_back(item.cast<float>());
                {
                    py::gil_scoped_release release;
                    p.callAddDelNodes(minF, maxF, junk);
                }
            },
            py::arg("minF"), py::arg("maxF"), py::arg("junk")=0
        )

        // --- Pickle -------------------------------------------------------
        .def(py::pickle(
            [](const Population &p) {
                py::list ind_list;
                for (const auto& net : p.individuals)
                    ind_list.append(py::cast(net));
                return py::make_tuple(
                    p.ni, p.jn, p.jnf, p.pn, p.pnf, p.fractalJudgment,
                    p.bestFit, p.indicesElite, p.meanFitness, p.minFitness, ind_list
                );
            },
            [](py::tuple t) {
                if (t.size() != 11)
                    throw std::runtime_error("Invalid state for Population!");
                Population p(
                    0,
                    t[0].cast<unsigned int>(),
                    t[1].cast<unsigned int>(),
                    t[2].cast<unsigned int>(),
                    t[3].cast<unsigned int>(),
                    t[4].cast<unsigned int>(),
                    t[5].cast<bool>()
                );
                p.bestFit = t[6].cast<float>();
                p.indicesElite = t[7].cast<std::vector<int>>();
                p.meanFitness = t[8].cast<float>();
                p.minFitness = t[9].cast<float>();

                p.individuals.clear();
                py::list ind_list = t[10].cast<py::list>();
                for (auto item : ind_list)
                    p.individuals.push_back(item.cast<Network>());
                return p;
            }
        ))

        ;
}
