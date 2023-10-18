cmake_minimum_required(VERSION 3.14)
project(pypopsift)

set (PYBIND11_PYTHON_VERSION "3.10")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)

set(PopSift_BUILD_EXAMPLES OFF CACHE INTERNAL "")
FetchContent_Declare(
  popsift
  GIT_REPOSITORY git@github.com:braincorp/popsift.git
  GIT_TAG        1.0.0
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(popsift)

add_subdirectory(pybind11)

pybind11_add_module(pypopsift src/main.cpp src/popsift.cpp)
target_link_libraries(pypopsift PUBLIC popsift)

if (DEFINED OUTPUT_DIR)
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}")
    set_target_properties(pypopsift PROPERTIES LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${OUTPUT_DIR}")
endif()
