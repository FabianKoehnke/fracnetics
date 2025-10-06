cmake_minimum_required(VERSION 3.15)
project(fracnetics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Add git pybind11 submodule
add_subdirectory(extern/pybind11)

include_directories(include)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Build the Python module
pybind11_add_module(_core bindings/bindings.cpp)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fracnetics
)

install(TARGETS _core DESTINATION fracnetics)

target_compile_features(_core PUBLIC cxx_std_20)

