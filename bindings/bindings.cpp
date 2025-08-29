#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>
#include "../include/Network.hpp"
#include "../include/Population.hpp"
#include "../include/GymnasiumWrapper.hpp"

namespace py = pybind11;

PYBIND11_MODULE(fracnetics, m) {
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
    .def_readwrite("k_d", &Node::k_d);

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
    .def_readwrite("usedNodes", &Network::usedNodes);

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
        .def("accuracy", &Population::accuracy, py::arg("X"), py::arg("y"), py::arg("dMax"), py::arg("penalty"))
        .def("gymnasium", 
                [](Population &self,
                    py::object env,
                    int dMax,
                    int penalty,
                    int maxSteps,
                    int maxConsecutiveP) {
                        GymEnvWrapper wrapper(env);
                        self.gymnasium(wrapper,dMax,penalty,maxSteps,maxConsecutiveP);
                    },
                py::arg("env"), py::arg("dMax"), py::arg("penalty"), py::arg("maxSteps"), py::arg("maxConsecutiveP"))
        .def("tournamentSelection", &Population::tournamentSelection, py::arg("N"), py::arg("E"))
        .def("callEdgeMutation", &Population::callEdgeMutation, py::arg("probInnerNodes"), py::arg("probStartNode"))
        .def("callBoundaryMutationNormal", &Population::callBoundaryMutationNormal, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationUniform", &Population::callBoundaryMutationUniform, py::arg("probability"))
        .def("callBoundaryMutationNetworkSizeDependingSigma", &Population::callBoundaryMutationNetworkSizeDependingSigma, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationEdgeSizeDependingSigma", &Population::callBoundaryMutationEdgeSizeDependingSigma, py::arg("probability"), py::arg("sigma"))
        .def("callBoundaryMutationFractal", &Population::callBoundaryMutationFractal, py::arg("probability"), py::arg("minF"), py::arg("maxF"))
        .def("crossover", &Population::crossover, py::arg("probability"))
        .def("callAddDelNodes", &Population::callAddDelNodes, py::arg("minF"), py::arg("maxF"));
}

