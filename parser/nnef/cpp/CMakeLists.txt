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

project(nnef CXX)

# build information
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executables in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# nnef library
add_library(${PROJECT_NAME}
        include/cnnef.h
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
        src/cnnef.cpp
        )

# build interface include dir is used when this cmake is included into
# a larger project
# install interface include dir will be put into the generated cmake config file
# during install step
target_include_directories(${PROJECT_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PUBLIC $<INSTALL_INTERFACE:include>)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

target_link_libraries(${PROJECT_NAME})

# install the library
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# then the headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include DESTINATION .)

# generate and install cmake config file for find_package
install(EXPORT ${PROJECT_NAME} DESTINATION lib/cmake/${PROJECT_NAME})

# generate an auxiliary config file also needed by find_package
# it just includes the previously generated nnef.cmake
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake "include(\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}.cmake)")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake DESTINATION lib/cmake/${PROJECT_NAME})
