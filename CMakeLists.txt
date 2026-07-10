cmake_minimum_required(VERSION 3.25)

project(pde_solvers VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(FetchContent)

option(PDE_ENABLE_MPI "Build the experimental MPI examples" OFF)
option(PDE_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(PDE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(ProjectOptions)

FetchContent_Declare(
  Eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL
  SYSTEM
)
set(PDE_BUILD_TESTING "${BUILD_TESTING}")
set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
FetchContent_MakeAvailable(Eigen)
set(BUILD_TESTING "${PDE_BUILD_TESTING}" CACHE BOOL "Build tests" FORCE)

add_library(pde_core INTERFACE)
add_library(pde::core ALIAS pde_core)
target_include_directories(
  pde_core INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_link_libraries(pde_core INTERFACE Eigen3::Eigen pde_project_options)
target_compile_features(pde_core INTERFACE cxx_std_20)

if(PDE_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)

  add_library(pde_mpi STATIC common/src/common_dummy.cpp)
  target_include_directories(pde_mpi PUBLIC common/include)
  target_link_libraries(pde_mpi PUBLIC pde::core MPI::MPI_CXX)

  add_executable(poisson_fd fd/poisson/poisson_main.cpp)
  target_link_libraries(poisson_fd PRIVATE pde_mpi)

  add_executable(fd_test_decomp fd/poisson/test_decomp.cpp)
  target_link_libraries(fd_test_decomp PRIVATE pde_mpi)
endif()

option(ENABLE_MFEM "Build MFEM-based modules" OFF)

if(ENABLE_MFEM)
  if(NOT PDE_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
  endif()
  add_subdirectory(mfem)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
