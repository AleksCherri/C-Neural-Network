cmake_minimum_required(VERSION 3.16)
project(neural-network C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Warnings
add_compile_options(-Wall -Wextra)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    add_compile_options(${OpenMP_C_FLAGS})
    link_libraries(${OpenMP_C_LIBS})
else()
    message(WARNING "OpenMP not found; parallel pragmas will be ignored")
endif()

# Optimization
add_compile_options(-O3 -march=native -flto -ffast-math)
add_link_options(-flto)   # LTO на этапе линковки

# Libs
link_libraries(m)

# Building
add_executable(main main.c mlp.c)
target_link_libraries(main PRIVATE OpenMP::OpenMP_C)