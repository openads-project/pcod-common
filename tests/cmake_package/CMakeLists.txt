cmake_minimum_required(VERSION 3.16)
project(pcod_common_package_test LANGUAGES CXX)

find_package(pcod_common 0.1 CONFIG REQUIRED)

add_executable(pcod_common_package_test main.cpp)
target_link_libraries(pcod_common_package_test PRIVATE pcod_common::pcod_common)

enable_testing()
add_test(NAME pcod_common_package_test COMMAND pcod_common_package_test)
