cmake_minimum_required(VERSION 3.15)
project(fracnetics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# use new pybind11 python finder
set(PYBIND11_FINDPYTHON ON)

# pybind11 as submodule
add_subdirectory(extern/pybind11)

# unified Python finder
find_package(Python REQUIRED COMPONENTS Interpreter Development)

include_directories(include)

# -------------------
# Python extension
# -------------------
pybind11_add_module(_core bindings/bindings.cpp)

set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fracnetics
)

target_link_libraries(_core PRIVATE Python::Python)

install(TARGETS _core DESTINATION fracnetics)

target_compile_features(_core PUBLIC cxx_std_20)
# -------------------
# Tests
# -------------------
find_package(GTest REQUIRED)

add_library(test_lib
    tests/crossover.cpp
    tests/network.cpp
)

target_link_libraries(test_lib
    PRIVATE
        pybind11::module
        Python::Python
        GTest::gtest
)

add_executable(runTests
    tests/crossover.cpp
    tests/network.cpp
)

target_link_libraries(runTests
    PRIVATE
        test_lib
        GTest::gtest_main
        pybind11::module
        Python::Python
)

enable_testing()
include(GoogleTest)
gtest_discover_tests(runTests)

