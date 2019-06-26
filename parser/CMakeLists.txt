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

# build information
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executables in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

add_definitions(-std=c++11)

# nnef library
add_library(nnef
        include/nnef.h
        include/nnef/common/binary.h
        include/nnef/common/dictionary.h
        include/nnef/common/error.h
        include/nnef/common/lexer.h
        include/nnef/common/parser.h
        include/nnef/common/prototype.h
        include/nnef/common/shapes.h
        include/nnef/common/typespec.h
        include/nnef/common/typeutils.h
        include/nnef/common/value.h
        include/nnef/comp/comp_parser.h
        include/nnef/comp/evaluation.h
        include/nnef/comp/expression.h
        include/nnef/comp/fragment.h
        include/nnef/comp/stdlib_source.h
        include/nnef/flat/flat_parser.h
        include/nnef/flat/quant_parser.h
        include/nnef/flat/stdlib_protos.h
        src/nnef.cpp
        )
target_include_directories(nnef PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# nnef-validator application
add_executable(nnef-validator src/main.cpp)
target_link_libraries(nnef-validator PUBLIC nnef)
