add_executable(run_tests run_tests.cpp)
target_link_libraries(run_tests
    PRIVATE
        benes_network_general
        benes_network_symmetric
)
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
