cmake_minimum_required(VERSION 3.16)
project(tabu_im_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tabu_im
    src/main.cpp
    src/shared.cpp
    src/celf.cpp
    src/tabu_helpers.cpp
    src/tabu_v3.cpp
)

target_include_directories(tabu_im PRIVATE include)

if(MSVC)
    target_compile_options(tabu_im PRIVATE /O2 /W4)
else()
    target_compile_options(tabu_im PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
