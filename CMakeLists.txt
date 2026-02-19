cmake_minimum_required(VERSION 3.16)

project(pde_solvers LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for VS Code IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type (if not provided)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Warnings (good habit) ---
add_library(project_warnings INTERFACE)
target_compile_options(project_warnings INTERFACE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# --- MPI ---
find_package(MPI REQUIRED)

# --- Common library (empty for now, but will host Decomp2D/Field/Halo/CG soon) ---
add_library(common STATIC
  common/src/common_dummy.cpp
)
target_include_directories(common PUBLIC common/include)
target_link_libraries(common PUBLIC project_warnings MPI::MPI_CXX)

# --- FD Poisson executable (just a hello for now) ---
add_executable(poisson_fd
  fd/poisson/poisson_main.cpp
)
target_link_libraries(poisson_fd PRIVATE common MPI::MPI_CXX)



# -------------------------------
# Optional MFEM modules
# -------------------------------
option(ENABLE_MFEM "Build MFEM-based modules" OFF)

if(ENABLE_MFEM)
  add_subdirectory(mfem)
endif()