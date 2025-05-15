# Copyright (c) 2017-2025 The Khronos Group Inc.
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

cmake_minimum_required(VERSION 3.10)

project(skriptnd CXX)

# build information
message(STATUS "Build Configuration: ${CMAKE_BUILD_TYPE}")
message(STATUS "Build executables in: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# skriptnd library
add_library(${PROJECT_NAME}
        include/skriptnd.h
        include/core/either.h
        include/core/error.h
        include/core/function.h
        include/core/maybe.h
        include/core/ordereddict.h
        include/core/packable.h
        include/core/position.h
        include/core/print.h
        include/core/result.h
        include/core/types.h
        include/core/value.h
        include/core/variant.h
        include/frontend/astexpr.h
        include/frontend/lexer.h
        include/frontend/operator.h
        include/frontend/parser.h
        include/frontend/polynom.h
        include/frontend/symbolic.h
        include/frontend/typing.h
        include/composer/composer.h
        include/composer/evaluation.h
        include/composer/simplification.h
        include/composer/model.h
        include/composer/tensorref.h
        include/composer/valuexpr.h
        src/skriptnd.cpp
        )

# build interface include dir is used when this cmake is included into
# a larger project
# install interface include dir will be put into the generated cmake config file
# during install step
target_include_directories(${PROJECT_NAME}
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/core>
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/frontend>
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/composer>
        PUBLIC $<INSTALL_INTERFACE:include>
        PUBLIC $<INSTALL_INTERFACE:include/core>
        PUBLIC $<INSTALL_INTERFACE:include/frontend>
        PUBLIC $<INSTALL_INTERFACE:include/composer>)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
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
# it just includes the previously generated skriptnd.cmake
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake "include(\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}.cmake)")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake DESTINATION lib/cmake/${PROJECT_NAME})
