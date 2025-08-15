#include <pybind11/pybind11.h>
#include "../src/main.cpp"

// C++ function to expose
double add(double a, double b) {
    return a + b;
}

// Define the Python module and bind the function
PYBIND11_MODULE(fracnetics, m) {
    m.doc() = "fracnetics";
    m.def("add", &add, "Add two numbers");
    m.def("run", run, "run evolution");
}

