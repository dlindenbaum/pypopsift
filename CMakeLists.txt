cmake_minimum_required(VERSION 3.14)
project(pypopsift)

set (PYBIND11_PYTHON_VERSION "3.10")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PopSift_BUILD_EXAMPLES OFF CACHE INTERNAL "")

add_subdirectory(pybind11)
add_subdirectory(popsift)

pybind11_add_module(pypopsift src/main.cpp src/popsift.cpp)
target_link_libraries(pypopsift PUBLIC popsift)

if (DEFINED OUTPUT_DIR)
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIR}")
endif()
