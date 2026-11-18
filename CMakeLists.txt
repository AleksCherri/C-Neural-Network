cmake_minimum_required(VERSION 3.16)
project(neural-network VERSION 0.1 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(mlp STATIC mlp.c)
target_include_directories(mlp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mlp PUBLIC m)

find_package(OpenMP QUIET)
if(OpenMP_C_FOUND)
  target_link_libraries(mlp PUBLIC OpenMP::OpenMP_C)
else()
  message(WARNING "OpenMP not found — parallel pragmas will be ignored")
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mlp PRIVATE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-march=native>
    $<$<CONFIG:Release>:-flto>)
  target_link_options(mlp PRIVATE $<$<CONFIG:Release>:-flto>)
endif()

add_executable(main main.c)
target_link_libraries(main PRIVATE mlp)

# enable_testing()
# add_executable(test_mlp tests/test_mlp.c)
# target_link_libraries(test_mlp PRIVATE mlp)
# add_test(NAME mlp COMMAND test_mlp)
