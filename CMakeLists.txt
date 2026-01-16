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

if(PCOD_COMMON_BUILD_TESTS)
  enable_testing()
  add_executable(pcod_common_test_nms tests/test_nms.cpp)
  target_link_libraries(pcod_common_test_nms PRIVATE pcod_common)
  add_test(NAME pcod_common_test_nms COMMAND pcod_common_test_nms)

  add_executable(pcod_common_test_manifest tests/test_manifest.cpp)
  target_link_libraries(pcod_common_test_manifest PRIVATE pcod_common)
  add_test(NAME pcod_common_test_manifest COMMAND pcod_common_test_manifest)

  add_executable(pcod_common_test_preprocess tests/test_point_preprocess.cpp)
  target_link_libraries(pcod_common_test_preprocess PRIVATE pcod_common)
  add_test(NAME pcod_common_test_preprocess COMMAND pcod_common_test_preprocess)

  add_executable(pcod_common_test_pbod_postprocess tests/test_pbod_postprocess.cpp)
  target_link_libraries(pcod_common_test_pbod_postprocess PRIVATE pcod_common)
  add_test(NAME pcod_common_test_pbod_postprocess COMMAND pcod_common_test_pbod_postprocess)
endif()

install(TARGETS pcod_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/pcod_common DESTINATION include)
