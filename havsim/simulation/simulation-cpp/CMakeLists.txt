cmake_minimum_required(VERSION 3.16)
project(havsim_cpp)

set(CMAKE_CXX_STANDARD 11)

add_library(havsim_cpp vehicle.cpp vehicle.h lane.cpp lane.h)