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

// Prevent pybind11 from deep-copying the entire std::vector<Network> on every
// Python access to pop.individuals.  With this macro the vector is treated as
// an opaque type and py::bind_vector provides a thin, reference-based wrapper
// instead.  This is the single biggest source of memory savings: without it,
// every `for ind in pop.individuals` copied ALL Network objects (including
// their node trees, edges, boundaries, decisions …).
PYBIND11_MAKE_OPAQUE(std::vector<Network>)

// Helper: fill a reusable vec2d buffer from a numpy float32 array.
// Uses a thread_local static buffer to avoid heap allocation/deallocation
// churn that contributes to memory fragmentation over many generations.
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

// Helper: invoke Python's gc.collect() to reclaim cyclic garbage.
// Called after heavy operations that create many temporary Python objects
// (gymnasium env.step/reset, data conversion, etc.) to prevent memory
// accumulation across generations.
static void force_gc_collect() {
    py::module_::import("gc").attr("collect")();
}

PYBIND11_MODULE(_core, m) {

    // Node
    py::class_<Node>(m, "Node")
    .def(py::init<
            std::shared_ptr<std::mt19937_64>,
            unsigned int,
            std::string,
            unsigned int>(),
         py::arg("generator"), py::arg("id"), py::arg("type"), py::arg("f"))
    .def_readwrite("id", &Node::id)
    .def_readwrite("type", &Node::type)
    .def_readwrite("f", &Node::f)
    .def_readwrite("edges", &Node::edges)
    .def_readwrite("boundaries", &Node::boundaries)
    .def_readwrite("productionRuleParameter", &Node::productionRuleParameter)
    .def_readwrite("k_d", &Node::k_d)
    .def_readwrite("used", &Node::used)
    .def_readwrite("traverseCounter", &Node::traverseCounter)
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
                n.used
            );
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 8)
                throw std::runtime_error("Invalid state for Node!");

            Node n(
                std::make_shared<std::mt19937_64>(std::random_device{}()), // new generator
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
    .def_readwrite("innerNodes", &Network::innerNodes)
    .def_readwrite("startNode", &Network::startNode)
    .def_readwrite("fitness", &Network::fitness)
    .def_readwrite("fitnessValues", &Network::fitnessValues)
    .def_readwrite("objectives", &Network::objectives)       // Pareto objectives
    .def_readwrite("lastStepRewards", &Network::lastStepRewards)
    .def_readwrite("decisions", &Network::decisions)
    .def_readwrite("currentNodeID", &Network::currentNodeID)
    .def_readwrite("invalid", &Network::invalid)
    .def_readwrite("nBest", &Network::nBest)
    .def_readwrite("nConsecutiveP", &Network::nConsecutiveP)
    .def_readwrite("nCrossovers", &Network::nCrossovers)
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
        // Pickle support – fixed: tuple has 12 elements (indices 0-11)
    .def(py::pickle(
        [](const Network &n) { // __getstate__
            return py::make_tuple(
                n.jn, n.jnf, n.pn, n.pnf, n.fractalJudgment,
                n.innerNodes, n.startNode, n.fitness, n.decisions,
                n.fitnessValues, n.objectives, n.lastStepRewards
            );
        },
        [](py::tuple t) { // __setstate__
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

    // Opaque vector binding – gives Python list-like access by reference,
    // no deep copies.  Pickle serialises to/from a plain Python list of
    // Network objects.
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
                std::vector<int>
                >(),
             py::arg("seed"), py::arg("ni"), py::arg("jn"), py::arg("jnf"),
             py::arg("pn"), py::arg("pnf"), py::arg("fractalJudgment"), py::arg("nFeatureValues"))
        .def_readonly("ni", &Population::ni)
        .def_readwrite("jn", &Population::jn)
        .def_readwrite("jnf", &Population::jnf)
        .def_readwrite("pn", &Population::pn)
        .def_readwrite("pnf", &Population::pnf)
        .def_readwrite("fractalJudgment", &Population::fractalJudgment)
        .def_readwrite("bestFit", &Population::bestFit)
        .def_readwrite("indicesElite", &Population::indicesElite)
        .def_readwrite("meanFitness", &Population::meanFitness)
        .def_readwrite("minFitness", &Population::minFitness)
        .def_readwrite("maxNetworkSize", &Population::maxNetworkSize)
        // Use def_property with return_value_policy::reference instead of
        // def_readwrite (which uses reference_internal / keep_alive).
        // reference_internal calls add_patient() on every property access,
        // accumulating Py_INCREF entries in pybind11's internals.patients map.
        // With plain reference, the wrapper simply points to the member
        // without adding keep_alive bookkeeping each time.
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
                    int dMax,
                    int maxSteps,
                    int maxConsecutiveP,
                    int worstFitness,
                    int seed
                    ) {
                        GymEnvWrapper wrapper(env);
                        self.gymnasium(wrapper, dMax, maxSteps, maxConsecutiveP, worstFitness, seed);
                        // Force GC to reclaim cyclic garbage from env.step()/env.reset()
                        // calls that accumulate over the population loop.
                        force_gc_collect();
                    },
                py::arg("env"), py::arg("dMax"), py::arg("maxSteps"), py::arg("maxConsecutiveP"), py::arg("worstFitness"), py::arg("seed")
            )

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
                        self.gymnasiumMultiSeed(wrapper, dMax, maxSteps, maxConsecutiveP, worstFitness, seeds);
                        // Force GC to reclaim cyclic garbage from env.step()/env.reset()
                        // calls that accumulate over population × seeds loops.
                        force_gc_collect();
                    },
                py::arg("env"), py::arg("dMax"), py::arg("maxSteps"), py::arg("maxConsecutiveP"), py::arg("worstFitness"), py::arg("seeds")
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
             py::arg("probInnerNodes"), py::arg("probStartNode"), py::arg("justUsedNodes")=false, py::arg("adaptToEdgeSize")=false)
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

                for (auto item : minF_py)
                    minF.push_back(item.cast<float>());

                for (auto item : maxF_py)
                    maxF.push_back(item.cast<float>());

                {
                    py::gil_scoped_release release;
                    p.callBoundaryMutationFractal(probability, minF, maxF, justUsedNodes);
                }
            },
            py::arg("probability"),
            py::arg("minF"),
            py::arg("maxF"),
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

                for (auto item : minF_py)
                    minF.push_back(item.cast<float>());

                for (auto item : maxF_py)
                    maxF.push_back(item.cast<float>());

                {
                    py::gil_scoped_release release;
                    p.callAddDelNodes(minF, maxF, junk);
                }
            },
            py::arg("minF"),
            py::arg("maxF"),
            py::arg("junk")=0
        )

        // pickle support – serialise individuals as a plain Python list
        .def(py::pickle(
        [](const Population &p) { // __getstate__
            py::list ind_list;
            for (const auto& net : p.individuals)
                ind_list.append(py::cast(net));
            return py::make_tuple(
                p.ni, p.jn, p.jnf, p.pn, p.pnf, p.fractalJudgment,
                p.bestFit, p.indicesElite, p.meanFitness, p.minFitness, ind_list
            );
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 11)
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
