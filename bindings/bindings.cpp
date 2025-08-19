#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "../include/Network.hpp"
#include "../include/Population.hpp"

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
        .def("callFitness", &Population::callFitness,
                py::arg("X"), py::arg("y"), py::arg("dMax"), py::arg("penalty"), py::arg("type"), py::arg("maxConsecutiveP"));
}

