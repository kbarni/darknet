cmake_minimum_required(VERSION 3.5)

project(examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(../include/ ${OpenCV_INCLUDE_DIRS})

add_executable(classification classification.cpp)
add_executable(detection detection.cpp)
add_executable(segmentation segmentation.cpp)

target_link_libraries(classification ${OpenCV_LIBS} -ldarknet)
target_link_libraries(detection ${OpenCV_LIBS} -ldarknet)
target_link_libraries(segmentation ${OpenCV_LIBS} -ldarknet)
