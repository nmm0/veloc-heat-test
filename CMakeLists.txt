cmake_minimum_required(VERSION 3.14)
project(heatdis)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

add_executable(heatdis_resil)
add_executable(heatdis_veloc)
add_executable(heatdis_synthesis_benchmark)

add_subdirectory(src)

find_package(MPI REQUIRED)
target_link_libraries(heatdis_resil PRIVATE MPI::MPI_CXX)
target_link_libraries(heatdis_veloc PRIVATE MPI::MPI_CXX)
target_link_libraries(heatdis_synthesis_benchmark PRIVATE MPI::MPI_CXX)

find_package(Kokkos REQUIRED)
target_link_libraries(heatdis_resil PRIVATE Kokkos::kokkos)
target_link_libraries(heatdis_veloc PRIVATE Kokkos::kokkos)
target_link_libraries(heatdis_synthesis_benchmark PRIVATE Kokkos::kokkos)

find_package(resilience REQUIRED)
target_link_libraries(heatdis_resil PRIVATE Kokkos::resilience)
target_link_libraries(heatdis_synthesis_benchmark PRIVATE Kokkos::resilience)

find_package(VeloC REQUIRED)
target_link_libraries(heatdis_veloc PRIVATE veloc::veloc)
target_link_libraries(heatdis_synthesis_benchmark PRIVATE veloc::veloc)

add_subdirectory(tpl)

# Install rules
include(GNUInstallDirs)

install(TARGETS heatdis_resil heatdis_veloc heatdis_synthesis_benchmark)
