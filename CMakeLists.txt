cmake_minimum_required(VERSION 3.15)
project(GNPfrac LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Add git pybind11 submodule
add_subdirectory(extern/pybind11)

include_directories(include)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Build the Python module
pybind11_add_module(fracnetics bindings/bindings.cpp)

set_target_properties(fracnetics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fracnetics
)

target_compile_features(fracnetics PUBLIC cxx_std_20)

