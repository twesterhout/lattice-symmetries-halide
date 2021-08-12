cmake_minimum_required(VERSION 3.16)
project(LatticeSymmetriesHalide)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_EXPORT_COMPILE_COMMANDS YES)

# Find Halide
find_package(Halide REQUIRED)
find_package(PkgConfig REQUIRED) 

pkg_check_modules(LatticeSymmetries REQUIRED IMPORTED_TARGET lattice_symmetries)

# Generator
add_executable(generator generator.cpp)
target_link_libraries(generator PRIVATE Halide::Generator)

add_halide_library(benes_network_general
    FROM generator
    GENERATOR benes_network_generator
    PARAMS spin_inversion=0
    # FEATURES no_asserts no_bounds_query
)
add_halide_library(benes_network_symmetric
    FROM generator
    GENERATOR benes_network_generator
    PARAMS spin_inversion=1
    # FEATURES no_asserts no_bounds_query
)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
    PRIVATE benes_network_general benes_network_symmetric
    PUBLIC  PkgConfig::LatticeSymmetries
)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
