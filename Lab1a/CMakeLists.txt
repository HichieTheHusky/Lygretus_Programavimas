cmake_minimum_required(VERSION 3.17)
project(Lab1)

set(CMAKE_CXX_STANDARD 17)

add_executable(Lab1 main.cpp)

#target_link_libraries(demo pthread)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(Lab1 Threads::Threads)

find_package(nlohmann_json REQUIRED)
target_link_libraries(Lab1 nlohmann_json::nlohmann_json)