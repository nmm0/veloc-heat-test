cmake_minimum_required(VERSION 3.17)
project(ABFT3D)

# Put the location of Kokkos
set(CMAKE_PREFIX_PATH /Users/knteran/ASC/KokkosBuild/install)
set(CMAKE_CXX_STANDARD 14)

add_executable(ABFT3D main.cpp)
find_package(Kokkos)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
target_link_libraries(ABFT3D PUBLIC Kokkos::kokkos)
