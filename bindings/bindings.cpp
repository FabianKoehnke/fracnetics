#include <pybind11/pybind11.h>

namespace py = pybind11;

// C++ function to expose
double add(double a, double b) {
    return a + b;
}

// Define the Python module and bind the function
PYBIND11_MODULE(_fractalgenetics, m) {
    m.doc() = "Test module with simple add function";
    m.def("add", &add, "Add two numbers");
}

