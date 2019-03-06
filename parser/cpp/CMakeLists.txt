# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.0)

project(nnef-validator CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build information
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executables in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

add_definitions(-std=c++11)

# Source directories
set(LIB_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(LIB_FLAT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flat)
set(LIB_COMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/comp)

# Include directories
include_directories( ${LIB_COMMON_DIR} ${LIB_FLAT_DIR} ${LIB_COMP_DIR} )

# The library is only composed of headers
file(GLOB LIB_HEADERS ${LIB_COMMON_DIR}/*.h ${LIB_FLAT_DIR}/*.h ${LIB_COMP_DIR}/*.h)


# Create the target
set(target nnef-validator)
add_executable(${target} ${LIB_HEADERS} main.cpp nnef.cpp)
