cmake_minimum_required(VERSION 3.16)

file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PCOD_COMMON_VERSION LIMIT_COUNT 1)
project(pcod_common VERSION "${PCOD_COMMON_VERSION}" LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

option(PCOD_COMMON_BUILD_TESTS "Build pcod-common tests" ON)

find_package(yaml-cpp QUIET CONFIG)
if(TARGET yaml-cpp::yaml-cpp)
  set(PCOD_COMMON_YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(PCOD_COMMON_YAML_CPP_TARGET yaml-cpp)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(YAML_CPP REQUIRED IMPORTED_TARGET yaml-cpp)
  set(PCOD_COMMON_YAML_CPP_TARGET PkgConfig::YAML_CPP)
endif()

option(PCOD_COMMON_ENABLE_CUDA "Build optional CUDA components for pcod-common" OFF)
set(PCOD_COMMON_CONFIG_WITH_CUDA OFF)

add_library(pcod_common STATIC
  src/bounding_box.cpp
  src/nms.cpp
  src/pbod_postprocess.cpp
  src/pillar_grid.cpp
  src/point_preprocess.cpp
  src/model_manifest.cpp
)
add_library(pcod_common::pcod_common ALIAS pcod_common)

set_target_properties(pcod_common PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

target_include_directories(pcod_common
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(pcod_common PUBLIC ${PCOD_COMMON_YAML_CPP_TARGET})

if(PCOD_COMMON_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(PCOD_COMMON_CONFIG_WITH_CUDA ON)
    target_sources(pcod_common PRIVATE src/pillar_preprocess_cuda_runtime.cu)
    target_link_libraries(pcod_common PRIVATE CUDA::cudart)
    set_source_files_properties(src/pillar_preprocess_cuda_runtime.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(pcod_common PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED YES
      CUDA_ARCHITECTURES "70;75;86;89;90;90-virtual"
      CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "CUDA compiler detected for pcod_common; building runtime capability probe")
  else()
    target_sources(pcod_common PRIVATE src/pillar_preprocess_cuda_stub.cpp)
    message(WARNING "PCOD_COMMON_ENABLE_CUDA=ON but no CUDA compiler was found; building CPU-only pcod_common")
  endif()
else()
  target_sources(pcod_common PRIVATE src/pillar_preprocess_cuda_stub.cpp)
endif()

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

set(PCOD_COMMON_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/pcod_common"
    CACHE STRING "Install directory for pcod-common CMake package files")
set(PCOD_COMMON_CONFIG_YAML_CPP_TARGET "${PCOD_COMMON_YAML_CPP_TARGET}")

configure_package_config_file(
  cmake/pcod_commonConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/pcod_commonConfig.cmake"
  INSTALL_DESTINATION "${PCOD_COMMON_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/pcod_commonConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

install(TARGETS pcod_common
  EXPORT pcod_commonTargets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(DIRECTORY include/pcod_common DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT pcod_commonTargets
  FILE pcod_commonTargets.cmake
  NAMESPACE pcod_common::
  DESTINATION "${PCOD_COMMON_INSTALL_CMAKEDIR}"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/pcod_commonConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/pcod_commonConfigVersion.cmake"
  DESTINATION "${PCOD_COMMON_INSTALL_CMAKEDIR}"
)
