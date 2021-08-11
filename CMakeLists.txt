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

add_halide_library(benes_network
    FROM generator GENERATOR benes_network_generator)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark
    PRIVATE benes_network
    PUBLIC  PkgConfig::LatticeSymmetries
)

# add_halide_library(experiment
#     FROM generator GENERATOR experimental_generator)

# add_executable(test_generator test_generator.cpp)
# target_link_libraries(test_generator
#     PRIVATE
#         compute_all
#         einstein_equations
#     # PRIVATE g_dd g_UU Dg_ddd Dg_dUU DDg_dddd
#     #         Gamma_Udd DGamma_dUdd
#     #         Xi_U Xi_d DXi_dU DXi_dd
#     # g_dd_ref g_UU_ref Dg_dd_ref Dg_UU_ref DDg_dd_ref Gamma_Udd_ref DGamma_dUdd_ref
# )
# target_include_directories(test_generator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
