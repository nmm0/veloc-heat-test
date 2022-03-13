cmake_minimum_required(VERSION 3.19)
project(heatdis)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

set(CMAKE_PREFIX_PATH /home/knteran/Kokkos_Haswell;/home/knteran/resilienceKT )
add_executable(heatdis)
add_executable(heatdis_resil)

add_subdirectory(src)

find_package(Kokkos REQUIRED)
target_link_libraries(heatdis PRIVATE Kokkos::kokkos)
target_link_libraries(heatdis_resil PRIVATE Kokkos::kokkos)

find_package(resilience REQUIRED)
target_link_libraries(heatdis_resil PRIVATE Kokkos::resilience)

target_compile_definitions(heatdis_resil PRIVATE USE_RESILIENT_EXEC)

add_subdirectory(tpl)

# Install rules
include(GNUInstallDirs)

install(TARGETS heatdis_resil heatdis)
