#include <pybind11/pybind11.h>
#include "../src/main.cpp"
#include "../include/Data.hpp"

// C++ function to expose
double add(double a, double b) {
    return a + b;
}

namespace py = pybind11;
// Define the Python module and bind the function
PYBIND11_MODULE(fracnetics, m) {
    m.doc() = "fracnetics";
    m.def("add", &add, "Add two numbers");
    py::class_<Population, std::shared_ptr<Population>>(m, "Population")
    .def(py::init<
            int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            unsigned int,
            bool>(),
         py::arg("seed"),
         py::arg("ni"),
         py::arg("jn"),
         py::arg("jnf"),
         py::arg("pn"),
         py::arg("pnf"),
         py::arg("fractalJudgment"))
    .def_readonly("ni", &Population::ni)
    .def_readwrite("jn", &Population::jn)
    .def_readwrite("jnf", &Population::jnf)
    .def_readwrite("pn", &Population::pn)
    .def_readwrite("pnf", &Population::pnf)
    .def_readwrite("fractalJudgment", &Population::fractalJudgment)
    .def_readwrite("bestFit", &Population::bestFit)
    .def_readwrite("meanFitness", &Population::meanFitness)
    .def_readwrite("minFitness", &Population::minFitness);

}

