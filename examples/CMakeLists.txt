cmake_minimum_required(VERSION 3.14)

project(LazyMathExamples CXX)

include(FetchContent)
FetchContent_Declare(LazyMath SOURCE_DIR "${PROJECT_SOURCE_DIR}/..")
FetchContent_MakeAvailable(LazyMath)

foreach(name IN ITEMS example-linear-solver example-constrained-linear-solver example-minimizer example-constrained-minimizer)
  add_executable("${name}" "${name}.cc")
  target_link_libraries("${name}" PRIVATE LazyMath::LazyMath)
  target_compile_features("${name}" PRIVATE cxx_std_17)
endforeach()
