cmake_minimum_required(VERSION 3.15)
project(TensorLLM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main library (header-only)
add_library(tensor INTERFACE)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(main tensor.cpp)
target_link_libraries(main PRIVATE tensor)

# Tiny transformer-style generation demo
add_executable(toy_llm toy_llm.cpp)
target_link_libraries(toy_llm PRIVATE tensor)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(main PRIVATE TENSOR_OPENMP_ENABLED=1)
	target_link_libraries(toy_llm PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(toy_llm PRIVATE TENSOR_OPENMP_ENABLED=1)
endif()

# Testing
enable_testing()

# Simple test executable (no external dependencies)
add_executable(tensor_tests tests/simple_test.cpp)
target_link_libraries(tensor_tests PRIVATE tensor)
target_compile_options(tensor_tests PRIVATE -Wall -Wextra)
if(OpenMP_CXX_FOUND)
	target_link_libraries(tensor_tests PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(tensor_tests PRIVATE TENSOR_OPENMP_ENABLED=1)
endif()

add_executable(toy_llm_tests tests/toy_llm_test.cpp)
target_link_libraries(toy_llm_tests PRIVATE tensor)
target_compile_options(toy_llm_tests PRIVATE -Wall -Wextra)
if(OpenMP_CXX_FOUND)
	target_link_libraries(toy_llm_tests PRIVATE OpenMP::OpenMP_CXX)
	target_compile_definitions(toy_llm_tests PRIVATE TENSOR_OPENMP_ENABLED=1)
endif()

# Register tests
add_test(NAME TensorTests COMMAND tensor_tests)
add_test(NAME ToyLLMTests COMMAND toy_llm_tests)
