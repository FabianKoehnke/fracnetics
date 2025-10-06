#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include <utility>
#include "../include/Network.hpp"
#include "../include/Population.hpp"
#include "../include/GymnasiumWrapper.hpp"

namespace py = pybind11;

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
    .def_readwrite("decisions", &Network::decisions)
    .def("traversePath", &Network::traversePath, py::arg("X"), py::arg("dMax"))
        // Pickle support
    .def(py::pickle(
        [](const Network &n) { // __getstate__
            return py::make_tuple(
                n.jn, n.jnf, n.pn, n.pnf, n.fractalJudgment,
                n.innerNodes, n.startNode, n.fitness, n.decisions
            );
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 9)
                throw std::runtime_error("Invalid state for Network!");

            Network net(
                std::make_shared<std::mt19937_64>(std::random_device{}()), // new generator 
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

            return net;
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
                bool>(),
             py::arg("seed"), py::arg("ni"), py::arg("jn"), py::arg("jnf"),
             py::arg("pn"), py::arg("pnf"), py::arg("fractalJudgment"))
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
        .def_readwrite("individuals", &Population::individuals)
        // Functions
        .def("setAllNodeBoundaries", &Population::setAllNodeBoundaries, py::arg("minF"), py::arg("maxF"))
        .def("callTraversePath", &Population::callTraversePath, py::arg("X"), py::arg("dMax"))
        .def("accuracy", &Population::accuracy, py::arg("X"), py::arg("y"), py::arg("dMax"), py::arg("penalty"))
        .def("gymnasium", 
                [](Population &self,
                    py::object env,
                    int dMax,
                    int penalty,
                    int maxSteps,
                    int maxConsecutiveP,
                    int worstFitness,
                    int seed) {
                        GymEnvWrapper wrapper(env);
                        self.gymnasium(wrapper,dMax,penalty,maxSteps,maxConsecutiveP,worstFitness,seed);
                    },
                py::arg("env"), py::arg("dMax"), py::arg("penalty"), py::arg("maxSteps"), py::arg("maxConsecutiveP"), py::arg("worstFitness"), py::arg("seed"))
        .def("tournamentSelection", &Population::tournamentSelection, py::arg("N"), py::arg("E"))
        .def("callEdgeMutation", &Population::callEdgeMutation, py::arg("probInnerNodes"), py::arg("probStartNode"))
        .def("callBoundaryMutationNormal", &Population::callBoundaryMutationNormal, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationUniform", &Population::callBoundaryMutationUniform, py::arg("probability"))
        .def("callBoundaryMutationNetworkSizeDependingSigma", &Population::callBoundaryMutationNetworkSizeDependingSigma, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationEdgeSizeDependingSigma", &Population::callBoundaryMutationEdgeSizeDependingSigma, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationFractal", &Population::callBoundaryMutationFractal, py::arg("probability"), py::arg("minF"), py::arg("maxF"))
        .def("crossover", &Population::crossover, py::arg("probability"))
        .def("callAddDelNodes", &Population::callAddDelNodes, py::arg("minF"), py::arg("maxF"))
        // pickle support 
        .def(py::pickle(
        [](const Population &p) { // __getstate__
            return py::make_tuple(
                p.ni, p.jn, p.jnf, p.pn, p.pnf, p.fractalJudgment,
                p.bestFit, p.indicesElite, p.meanFitness, p.minFitness, p.individuals
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
            p.individuals = t[10].cast<std::vector<Network>>(); // passe hier den Typ deiner Individuals an

            return p;
        }
    ))


        ;
}

