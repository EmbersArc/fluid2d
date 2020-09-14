cmake_minimum_required(VERSION 3.5)
project(fluid)

set(CMAKE_CXX_STANDARD 17)

set(DEBUG_OPTIONS -Wall -Wextra -Wpedantic)
set(RELEASE_OPTIONS -O3)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(SFML COMPONENTS system window graphics REQUIRED)

add_executable(fluid fluid.cpp)
target_link_libraries(fluid Eigen3::Eigen sfml-system sfml-window sfml-graphics)

target_compile_options(fluid PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_OPTIONS}>")
target_compile_options(fluid PUBLIC "$<$<CONFIG:RELEASE>:${RELEASE_OPTIONS}>")
