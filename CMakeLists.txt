cmake_minimum_required(VERSION 3.11)
project(LatticeSymmetriesHalide)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

option(USE_SYSTEM_HALIDE OFF)

# Detect operating system for Halide targets
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(os_for_Halide "linux")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(os_for_Halide "osx")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(os_for_Halide "windows")
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(bits_for_Halide 64)
else() 
    set(bits_for_Halide 32)
endif()

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86|AMD64)")
    set(arch_for_Halide x86)
else()
    set(arch_for_Halide arm)
endif()

# Find Halide
if(USE_SYSTEM_HALIDE)
  find_package(Halide REQUIRED)
else()
  include(FetchContent)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
      set(_archive_suffix "zip")
  else()
      set(_archive_suffix "tar.gz")
  endif()
  FetchContent_Declare(
    HalideBinaryRelease
    URL https://github.com/halide/Halide/releases/download/v12.0.1/Halide-12.0.1-${arch_for_Halide}-${bits_for_Halide}-${os_for_Halide}-5dabcaa9effca1067f907f6c8ea212f3d2b1d99a.${_archive_suffix}
  )
  FetchContent_GetProperties(HalideBinaryRelease)
  if(NOT halidebinaryrelease_POPULATED)
    message(STATUS "[lattice-symmetries] Downloading binary release of Halide. This may take a while...")
    FetchContent_Populate(HalideBinaryRelease)
  endif()
  FetchContent_GetProperties(HalideBinaryRelease)
  find_package(HalideHelpers REQUIRED
      PATHS ${halidebinaryrelease_SOURCE_DIR}/lib/cmake)
  find_package(Halide REQUIRED
      PATHS ${halidebinaryrelease_SOURCE_DIR}/lib/cmake)
endif()

find_package(PkgConfig REQUIRED) 

pkg_check_modules(LatticeSymmetries REQUIRED IMPORTED_TARGET lattice_symmetries)

# Generator
add_executable(generator generator.cpp)
target_link_libraries(generator PRIVATE Halide::Generator)

function(generate_kernels_for_arch)
  set(_options)
  set(_oneValueArgs FROM GENERATOR BASE_NAME BASE_ARCH OUTPUT_LIST)
  set(_multiValueArgs SUFFIXES FEATURES PARAMS)
  cmake_parse_arguments(generate_kernels_for_arch
      "${_options}" "${_oneValueArgs}" "${_multiValueArgs}" ${ARGN})
  set(_start 0)
  list(LENGTH generate_kernels_for_arch_SUFFIXES _stop)

  # This is an ugly hack to handle empty SUFFIXES and FEATURES lists. We want to
  # append empty strings to them but CMake does not allow that, so instead we
  # create two element lists and reset the loop counter
  if (_stop EQUAL 0)
    set(generate_kernels_for_arch_SUFFIXES "dummy")
    set(generate_kernels_for_arch_FEATURES "dummy")
    list(APPEND generate_kernels_for_arch_SUFFIXES "")
    list(APPEND generate_kernels_for_arch_FEATURES "")
    set(_start 1)
    set(_stop 1)
  else()
    list(APPEND generate_kernels_for_arch_SUFFIXES "")
    list(APPEND generate_kernels_for_arch_FEATURES "")
  endif()

  # set(_output_list "")
  foreach(_i RANGE ${_start} ${_stop})
    list(GET generate_kernels_for_arch_SUFFIXES ${_i} _suffix)
    list(GET generate_kernels_for_arch_FEATURES ${_i} _feature)
    set(_library "${generate_kernels_for_arch_BASE_NAME}${_suffix}")
    # message(STATUS "[lattice-symmetries] Will generate ${_suffix} kernel for ${_feature}")
    add_halide_library(${_library}
        FROM ${generate_kernels_for_arch_FROM}
        GENERATOR ${generate_kernels_for_arch_GENERATOR}
        PARAMS ${generate_kernels_for_arch_PARAMS}
        TARGETS ${generate_kernels_for_arch_BASE_ARCH}${_feature}
    )
    # USE_RUNTIME ${generate_kernels_for_arch_USE_RUNTIME}
    list(APPEND _output_list ${_library})
  endforeach()
  set(${generate_kernels_for_arch_OUTPUT_LIST} ${_output_list} PARENT_SCOPE)
endfunction()

if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(x86|AMD64)")
  set(x86_64_suffixes _sse41 _avx _avx2)
  set(x86_64_features -sse41 -avx-sse41 -avx-avx2-fma-sse41)

  generate_kernels_for_arch(
      FROM generator
      GENERATOR benes_network_generator
      PARAMS spin_inversion=0
      BASE_NAME ls_internal_state_info_general_kernel_64
      BASE_ARCH x86-64-${os_for_Halide}
      SUFFIXES ${x86_64_suffixes}
      FEATURES ${x86_64_features}
      OUTPUT_LIST general_kernels
  )
  generate_kernels_for_arch(
      FROM generator
      GENERATOR benes_network_generator
      PARAMS spin_inversion=1
      BASE_NAME ls_internal_state_info_symmetric_kernel_64
      BASE_ARCH x86-64-${os_for_Halide}
      SUFFIXES ${x86_64_suffixes}
      FEATURES ${x86_64_features}
      OUTPUT_LIST symmetric_kernels
  )
  generate_kernels_for_arch(
      FROM generator
      GENERATOR benes_network_generator
      PARAMS spin_inversion=-1
      BASE_NAME ls_internal_state_info_antisymmetric_kernel_64
      BASE_ARCH x86-64-${os_for_Halide}
      SUFFIXES ${x86_64_suffixes}
      FEATURES ${x86_64_features}
      OUTPUT_LIST antisymmetric_kernels
  )

  set(Kernels
      ${general_kernels}
      ${symmetric_kernels}
      ${antisymmetric_kernels}
  )
else()
  set(base_name ls_internal_state_info_general_kernel_64)
  set(base_arch host)

  add_halide_library(${base_name}
      FROM generator GENERATOR benes_network_generator
      PARAMS spin_inversion=0
      TARGETS ${base_arch})

  set(Kernels
      ${base_name}
  )
endif()

add_library(lattice_symmetries_kernels
    kernels.hpp kernels.cpp
)
target_link_libraries(lattice_symmetries_kernels
    PRIVATE ${Kernels} Halide::Runtime
    PUBLIC  PkgConfig::LatticeSymmetries
)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
    PUBLIC  lattice_symmetries_kernels
)
target_include_directories(benchmark PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
