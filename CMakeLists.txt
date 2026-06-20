cmake_minimum_required(VERSION 3.20)
project(MA-CUSSP LANGUAGES CXX)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost 1.65 REQUIRED CONFIG COMPONENTS log thread filesystem system)

file(GLOB_RECURSE CORE_SOURCES
    "src/*.cpp"
    "src/HighLevel/*.cpp"
    "src/LowLevel/*.cpp"
    "src/LowLevel/Utils/*.cpp"
    "src/InfoGathering/*.cpp"
    "src/pipeline/*.cpp"
)
list(REMOVE_ITEM CORE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/main.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline_main.cpp
)

add_library(mapf_core STATIC ${CORE_SOURCES})
target_include_directories(mapf_core PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_compile_options(mapf_core PRIVATE -Wall -g)
target_link_libraries(mapf_core PUBLIC
  Boost::log
  Boost::thread
  Boost::filesystem
  Boost::system
)

add_executable(macussp_solver src/main.cpp)
target_link_libraries(macussp_solver PRIVATE mapf_core)

add_executable(macussp_pipeline src/pipeline_main.cpp)
target_link_libraries(macussp_pipeline PRIVATE mapf_core)

enable_testing()

function(add_macussp_test name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/inc)
  target_compile_definitions(${name} PRIVATE MACUSSP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(${name} PRIVATE mapf_core)
  add_test(NAME ${name} COMMAND ${name} --fast WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

add_macussp_test(test_lex_compare tests/test_lex_compare.cpp)
add_macussp_test(test_belief_set tests/test_belief_set.cpp)
add_macussp_test(test_lastar tests/test_lastar.cpp)
add_macussp_test(test_apex_lastar_equiv tests/test_apex_lastar_equiv.cpp)
add_macussp_test(test_pipeline_golden tests/test_pipeline_golden.cpp)

add_executable(test_saia_sanity tests/test_saia_sanity.cpp)
target_include_directories(test_saia_sanity PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(test_saia_sanity PRIVATE mapf_core)
add_test(NAME test_saia_sanity COMMAND test_saia_sanity --fast
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test_landmark_sequence tests/test_landmark_sequence.cpp)
target_include_directories(test_landmark_sequence PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_compile_definitions(test_landmark_sequence PRIVATE MACUSSP_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(test_landmark_sequence PRIVATE mapf_core)
add_test(NAME test_landmark_sequence COMMAND test_landmark_sequence --fast
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test_env_costs tests/test_env_costs.cpp)
target_include_directories(test_env_costs PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(test_env_costs PRIVATE mapf_core)
add_test(NAME test_env_costs COMMAND test_env_costs --fast
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(test_baseline_fairness tests/test_baseline_fairness.cpp)
target_include_directories(test_baseline_fairness PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(test_baseline_fairness PRIVATE mapf_core)
# Slow gate (not in default CTest): ./bin/test_baseline_fairness

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
