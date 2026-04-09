cmake_minimum_required(VERSION 3.16)
project(pcod_common LANGUAGES CXX)

option(PCOD_COMMON_BUILD_TESTS "Build pcod-common tests" ON)

find_package(yaml-cpp REQUIRED)

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

target_link_libraries(pcod_common PUBLIC yaml-cpp)

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

  add_executable(pcod_common_test_pbod_postprocess_validation tests/test_pbod_postprocess_validation.cpp)
  target_link_libraries(pcod_common_test_pbod_postprocess_validation PRIVATE pcod_common)
  add_test(NAME pcod_common_test_pbod_postprocess_validation COMMAND pcod_common_test_pbod_postprocess_validation)

  add_executable(pcod_common_test_pillar_grid tests/test_pillar_grid.cpp)
  target_link_libraries(pcod_common_test_pillar_grid PRIVATE pcod_common)
  add_test(NAME pcod_common_test_pillar_grid COMMAND pcod_common_test_pillar_grid)

  add_executable(pcod_common_test_bounding_box tests/test_bounding_box.cpp)
  target_link_libraries(pcod_common_test_bounding_box PRIVATE pcod_common)
  add_test(NAME pcod_common_test_bounding_box COMMAND pcod_common_test_bounding_box)

  add_executable(pcod_common_test_manifest_validation tests/test_manifest_validation.cpp)
  target_link_libraries(pcod_common_test_manifest_validation PRIVATE pcod_common)
  add_test(NAME pcod_common_test_manifest_validation COMMAND pcod_common_test_manifest_validation)

  add_executable(pcod_common_test_py_cpp_contract tests/test_py_cpp_contract.cpp)
  target_link_libraries(pcod_common_test_py_cpp_contract PRIVATE pcod_common)
  target_compile_definitions(pcod_common_test_py_cpp_contract PRIVATE PCOD_COMMON_PYTHON_DIR="${CMAKE_CURRENT_SOURCE_DIR}/python")
  add_test(NAME pcod_common_test_py_cpp_contract COMMAND pcod_common_test_py_cpp_contract)

  add_executable(pcod_common_test_readme_example tests/test_readme_example.cpp)
  target_link_libraries(pcod_common_test_readme_example PRIVATE pcod_common)
  add_test(NAME pcod_common_test_readme_example COMMAND pcod_common_test_readme_example)
endif()

install(TARGETS pcod_common
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/pcod_common DESTINATION include)
