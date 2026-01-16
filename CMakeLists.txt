cmake_minimum_required(VERSION 3.16)
project(pcod_common LANGUAGES CXX)

option(PCOD_COMMON_BUILD_TESTS "Build pcod-common tests" ON)

add_library(pcod_common STATIC
  src/bounding_box.cpp
  src/nms.cpp
  src/pbod_postprocess.cpp
  src/pillar_grid.cpp
  src/point_preprocess.cpp
  src/model_manifest.cpp
)

set_target_properties(pcod_common PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

target_include_directories(pcod_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

find_package(yaml-cpp REQUIRED)
target_link_libraries(pcod_common PUBLIC yaml-cpp)

if(PCOD_COMMON_BUILD_TESTS)
  enable_testing()
  add_executable(pcod_common_tests tests/test_nms.cpp tests/test_manifest.cpp)
  target_link_libraries(pcod_common_tests PRIVATE pcod_common)
  add_test(NAME pcod_common_tests COMMAND pcod_common_tests)
endif()

install(TARGETS pcod_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/pcod_common DESTINATION include)
