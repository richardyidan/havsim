cmake_minimum_required(VERSION 3.16)

project(havsim_cpp)

set(CMAKE_CXX_STANDARD 11)

add_library(havsim_cpp vehicle.cpp vehicle.h lane.cpp lane.h)

target_include_directories (havsim_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable (test test.cpp)

target_link_libraries (test LINK_PUBLIC havsim_cpp)
