#ifndef GYMNASIUM_WRAPPER_HPP
#define GYMNASIUM_WRAPPER_HPP
#ifdef WITH_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class GymEnvWrapper {
private:
    py::object env;

public:
    GymEnvWrapper() = default;
    explicit GymEnvWrapper(const py::object& env_obj) : env(env_obj) {}

    py::tuple reset() {
        return env.attr("reset")();
    }

    py::tuple step(const py::object& action) {
        return env.attr("step")(action);
    }

    py::tuple step(int action) {
        return step(py::int_(action));
    }

    void render() {
        env.attr("render")();
    }

    void close() {
        env.attr("close")();
    }
};
#endif
#endif
